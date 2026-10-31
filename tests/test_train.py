import json

import pytest
import torch

from conftest import tiny_config
from llmlab.checkpoint import CKPT_NAME, load_checkpoint, save_atomic
from llmlab.train import lr_at, make_optimizer, train
from llmlab.model import build_model


def run(cfg, d, root, **kw):
    return train(cfg, d, data_root=root, device="cpu", log=lambda *_: None, **kw)


def test_lr_schedule():
    assert lr_at(0, 1.0, 0.1, 10, 100) == pytest.approx(0.1)
    assert lr_at(9, 1.0, 0.1, 10, 100) == pytest.approx(1.0)
    assert lr_at(100, 1.0, 0.1, 10, 100) == pytest.approx(0.1)
    assert 0.1 < lr_at(55, 1.0, 0.1, 10, 100) < 1.0


def test_param_groups_no_decay_for_norm_bias_embedding():
    m = build_model(tiny_config(bias=True).model)
    opt = make_optimizer(m, tiny_config().train, "cpu")
    dec, nodec = opt.param_groups
    assert dec["weight_decay"] > 0 and nodec["weight_decay"] == 0
    assert all(p.ndim >= 2 for p in dec["params"])
    assert not any(p is m.wte.weight or p is m.wpe.weight for p in dec["params"])
    assert any(p is m.wte.weight for p in nodec["params"])
    assert len(dec["params"]) + len(nodec["params"]) == len(list(m.parameters()))


def test_resume_equals_uninterrupted(corpus_root, tmp_path):
    cfg = tiny_config()
    full = run(cfg, tmp_path / "a", corpus_root)
    assert full["finished"]
    part = run(cfg, tmp_path / "b", corpus_root, until_step=4)
    assert not part["finished"] and part["step"] == 4
    resumed = run(cfg, tmp_path / "b", corpus_root)
    assert resumed["finished"] and resumed["step"] == 10 and resumed["run_id"] == part["run_id"]
    a = load_checkpoint(tmp_path / "a" / CKPT_NAME)
    b = load_checkpoint(tmp_path / "b" / CKPT_NAME)
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k
    assert (tmp_path / "a" / "metrics.json").exists()
    m = json.loads((tmp_path / "b" / "metrics.json").read_text())
    assert m["n_params"] > 0 and m["final_val_loss"] is not None and m["runtime"]["amp_dtype"] == "none"


def test_checkpoint_atomic(tmp_path, monkeypatch):
    p = tmp_path / "latest.pt"
    save_atomic(p, {"x": torch.ones(3)})
    assert not list(tmp_path.glob("*.tmp"))

    real = torch.save

    def broken(obj, f):
        f.write(b"partial")
        raise OSError("диск отвалился")

    monkeypatch.setattr(torch, "save", broken)
    with pytest.raises(OSError):
        save_atomic(p, {"x": torch.zeros(3)})
    monkeypatch.setattr(torch, "save", real)
    assert torch.equal(load_checkpoint(p)["x"], torch.ones(3))  # старый файл цел
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_contents(corpus_root, tmp_path):
    run(tiny_config(), tmp_path, corpus_root, until_step=2)
    ck = load_checkpoint(tmp_path / CKPT_NAME)
    for k in ("model", "optimizer", "scaler", "step", "rng", "config", "wandb_run_id"):
        assert k in ck
    assert {"python", "numpy", "torch", "batch_gen"} <= set(ck["rng"])


def test_compile_checkpoint_keys_without_orig_mod(corpus_root, tmp_path, monkeypatch):
    monkeypatch.setenv("LLMLAB_COMPILE_BACKEND", "eager")
    cfg = tiny_config()
    cfg.train.compile = "true"
    m = run(cfg, tmp_path, corpus_root, until_step=3)
    assert m["runtime"]["compile"] is True
    ck = load_checkpoint(tmp_path / CKPT_NAME)
    assert not any("_orig_mod" in k for k in ck["model"])
    build_model(cfg.model).load_state_dict(ck["model"])  # грузится в обычный модуль
    cfg2 = tiny_config()
    cfg2.train.compile = "false"
    assert run(cfg2, tmp_path, corpus_root)["finished"]  # продолжение без compile


def test_nan_stops_with_clear_error(corpus_root, tmp_path):
    cfg = tiny_config()
    cfg.train.lr = 1e30
    with pytest.raises(FloatingPointError, match="NaN|не конечен"):
        run(cfg, tmp_path, corpus_root)
