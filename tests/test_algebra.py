"""algebra_cot без сети: синтетический CSV, char-токенизатор, сплиты, компактное хранение, метрика точности."""
import csv
import json
import random
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import tiny_config
from llmlab import algebra
from llmlab.algebra import (check_no_overlap, dedup_and_split, example_prompt, example_text, normalize_instruction,
                            parse_instruction, read_csv)
from llmlab.config import Config, ModelConfig, load_config
from llmlab.data import (corpus_dir, load_corpus_tokenizer, load_tokens_tensor, make_batch_generator, prepare,
                         random_batch, window_batch)
from llmlab.metrics import evaluate_accuracy, parse_answer
from llmlab.model import build_model
from llmlab.tokenizer import CharTokenizer, UnknownCharError, eos_id, load_tokenizer
from llmlab.train import train


def make_rows(n: int, seed: int = 0) -> list[dict]:
    """Строки в трёх формах reasoning (A: начинается с instruction, B: первая строка - условие без '=',
    C: условия нет)."""
    rng = random.Random(seed)
    pairs = rng.sample([(a, b) for a in range(-60, 60) for b in range(-60, 60)], n)
    rows = []
    for i, (a, b) in enumerate(pairs):
        op = "+" if i % 2 else "-"
        ans = a + b if op == "+" else a - b
        ins, base = f"{a} {op} {b} = ", f"{a} {op} {b}"
        steps, last = "1 + 1 = 2\n3 + 4 = 7", f"{base} = {ans}"
        rea = [f"{ins}-({b} + {a})\n{steps}\n{last}", f"{base}\n{steps}\n{last}", f"0{base}\n{steps}\n{last}"][i % 3]
        rows.append({"instruction": ins, "answer": str(ans), "reasoning": rea})
    return rows


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["instruction", "answer", "reasoning"])
        w.writeheader()
        w.writerows(rows)


def algebra_config(tmp_path, vocab: int, **train) -> Config:
    d = {
        "name": "alg", "seed": 3,
        "data": {"dataset": "algebra_cot", "tokenizer": "char", "data_dir": str(tmp_path), "vocab_size": vocab,
                 "val_size": 20, "test_size": 20, "split_seed": 5, "train_file": "train.csv"},
        "model": {"block_size": 64, "n_layer": 2, "n_head": 2, "n_embd": 32, "dropout": 0.1},
        "train": {"batch_size": 8, "max_steps": 6, "lr": 3e-3, "warmup_steps": 2, "eval_interval": 3,
                  "eval_batches": 2, "log_interval": 2, "ckpt_interval": 3, "sample_interval": 6,
                  "sample_max_new_tokens": 8, "sample_prompts": ["1 + 2 = "], "amp_dtype": "none",
                  "compile": "false", "accuracy_interval": 3, "accuracy_samples": 6, "accuracy_final_samples": 10,
                  "accuracy_batch_size": 4, **train},
        "wandb": {"mode": "disabled"},
    }
    return Config.from_dict(d)


@pytest.fixture(scope="module")
def alg_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("alg")
    (root / "raw" / "algebra").mkdir(parents=True)
    write_csv(root / "raw" / "algebra" / "train.csv", make_rows(300))
    probe = algebra_config(root, 1)
    try:
        prepare(probe)
    except ValueError as e:  # узнаём реальный размер словаря по сообщению об ошибке
        assert "data.vocab_size" in str(e)
    texts = [example_text(r) for r in make_rows(300)]
    vocab = CharTokenizer.train(texts).get_vocab_size()
    cfg = algebra_config(root, vocab)
    prepare(cfg)
    return root, cfg


# --- разбор и текст -------------------------------------------------------------
def test_csv_multiline_and_text_forms(tmp_path):
    rows = make_rows(9)
    write_csv(tmp_path / "t.csv", rows)
    got = read_csv(tmp_path / "t.csv")
    assert got == rows and "\n" in got[0]["reasoning"]
    ins = rows[0]["instruction"]
    a, b, c = (example_text(r) for r in rows[:3])
    assert a == rows[0]["reasoning"] and not a[len(ins):].startswith(ins)  # A: в начале условие не повторяется
    assert b == rows[1]["instruction"] + "\n" + "\n".join(rows[1]["reasoning"].split("\n")[1:])  # B: повтор убран
    assert c == rows[2]["instruction"] + rows[2]["reasoning"]  # C
    for r in rows:
        t = example_text(r)
        assert t.startswith(example_prompt(r))  # подсказка - префикс обучающего текста
        assert t.split("\n")[-1].endswith(" = " + r["answer"])
    assert parse_instruction("-5 - -7 = ") == (-5, "-", -7)
    with pytest.raises(ValueError):
        parse_instruction("5 * 7 = ")


def test_empty_field_is_error(tmp_path):
    (tmp_path / "t.csv").write_text("instruction,answer,reasoning\n1 + 1 = ,,x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_csv(tmp_path / "t.csv")


# --- char-токенизатор -----------------------------------------------------------
def test_char_tokenizer_roundtrip_eos_and_unknown(tmp_path):
    tok = CharTokenizer.train(["12 + 3 = 15\n", "-(4)"])
    assert eos_id(tok) == 0 and tok.get_vocab_size() == len(tok.chars) + 1
    text = "12 + 3 = 15\n-(4)"
    ids = tok.encode(text).ids
    assert 0 not in ids and tok.decode(ids) == text
    assert tok.decode([0] + ids + [0]) == text  # EOS пропускается
    tok.save(tmp_path / "tokenizer.json")
    tok2 = load_tokenizer(tmp_path / "tokenizer.json")
    assert isinstance(tok2, CharTokenizer) and tok2.encode(text).ids == ids
    with pytest.raises(UnknownCharError, match="вне словаря"):
        tok.encode("12 * 3")
    with pytest.raises(UnknownCharError):
        tok.encode_stream(["12 * 3"])
    texts = ["12 + 3", "-(4)", "3 = 15\n"]
    stream = tok.encode_stream(texts, chunk=2)
    want = [i for t in texts for i in tok.encode(t).ids + [0]]
    assert stream.tolist() == want
    m = build_model(ModelConfig(vocab_size=tok.get_vocab_size(), n_embd=32, n_head=2, n_layer=1, block_size=32))
    assert m.padded_vocab == 64  # словарь по-прежнему дополняется до кратного 64


# --- сплиты ---------------------------------------------------------------------
def test_dedup_before_split_and_no_overlap():
    rows = make_rows(200)
    dups = [{**rows[3], "instruction": rows[3]["instruction"].replace(" ", "")}, dict(rows[10]), dict(rows[10])]
    mixed = rows[:50] + dups + rows[50:]
    train, val, test, rep = dedup_and_split(mixed, 15, 25, seed=9)
    assert rep["duplicates_removed"] == 3 and rep["unique"] == 200
    assert len(val) == 15 and len(test) == 25 and len(train) == 160
    key = lambda part: {normalize_instruction(r["instruction"]) for r in part}  # noqa: E731
    assert not key(train) & key(val) and not key(train) & key(test) and not key(val) & key(test)
    assert rep["overlap_instruction"] == {"train_val": 0, "train_test": 0, "val_test": 0}
    again = dedup_and_split(mixed, 15, 25, seed=9)
    assert [r["instruction"] for r in again[1]] == [r["instruction"] for r in val]
    other = dedup_and_split(mixed, 15, 25, seed=10)
    assert [r["instruction"] for r in other[1]] != [r["instruction"] for r in val]
    with pytest.raises(AssertionError):
        check_no_overlap(train, val[:1] + train[:1], test)


def test_prepare_algebra_files_and_val_split_ok(alg_root):
    root, cfg = alg_root
    cdir = corpus_dir(cfg)
    assert cdir.name == "algebra_cot_char" and cdir != corpus_dir(tiny_config())
    meta = json.loads((cdir / "meta.json").read_text())
    assert meta["dtype"] == "uint8" and meta["tokenizer"] == "char"
    arr = np.fromfile(cdir / "train.bin", dtype=np.uint8)
    assert arr.size == meta["splits"]["train"]["tokens"]
    assert (arr == 0).sum() == meta["splits"]["train"]["stories"] == 260  # EOS после каждого примера
    val = algebra.read_examples(cdir / "val.jsonl")
    test = algebra.read_examples(cdir / "test.jsonl")
    assert len(val) == 20 and len(test) == 20
    # val-токены: каждый пример = текст + EOS
    tok = load_corpus_tokenizer(cdir)
    first = tok.decode(np.fromfile(cdir / "val.bin", dtype=np.uint8)[: len(example_text(val[0]))].tolist())
    assert first == example_text(val[0])


def test_wrong_char_vocab_is_clear_error(tmp_path):
    (tmp_path / "raw" / "algebra").mkdir(parents=True)
    write_csv(tmp_path / "raw" / "algebra" / "train.csv", make_rows(100))
    with pytest.raises(ValueError, match="data.vocab_size"):
        prepare(algebra_config(tmp_path, 5))


# --- компактное хранение ----------------------------------------------------------
def test_compact_storage_gives_same_batches_as_int64(alg_root):
    root, cfg = alg_root
    cdir = corpus_dir(cfg)
    compact = load_tokens_tensor(cdir, "train", "cpu")
    assert compact.dtype == torch.uint8
    wide = compact.to(torch.int64)
    g1, g2 = make_batch_generator("cpu", 11), make_batch_generator("cpu", 11)
    for _ in range(3):
        x1, y1 = random_batch(compact, 32, 8, g1)
        x2, y2 = random_batch(wide, 32, 8, g2)
        assert x1.dtype == torch.long and torch.equal(x1, x2) and torch.equal(y1, y2)
    offs = torch.tensor([0, 5, 100])
    assert torch.equal(window_batch(compact, offs, 16)[0], window_batch(wide, offs, 16)[0])


def test_ru_corpus_still_int32(corpus_root):
    cfg = tiny_config()
    assert corpus_dir(cfg, corpus_root).name == "vocab300"
    assert load_tokens_tensor(corpus_dir(cfg, corpus_root), "train", "cpu").dtype == torch.int32


# --- метрика ----------------------------------------------------------------------
@pytest.mark.parametrize("text,want", [
    ("5 - 7 = \n5 - 7 = -2", -2), ("1 + 2 = 3", 3), ("x\n2 + 2 = 4\n", None), ("1 + 2 = ", None),
    ("1 + 2 = abc", None), ("1 + 2 = 3.5", None), ("1 + 2 =3", None), ("", None), ("9 - 3 = 0 = -0", 0),
    ("-4 - -4 = 00", 0), ("5 = - 3", None),
])
def test_parse_answer(text, want):
    assert parse_answer(text) == want


class OracleLM(nn.Module):
    """Заглушка: по префиксу выдаёт следующий символ эталонного текста (или EOS в конце)."""

    def __init__(self, tok, texts, block_size=64):
        super().__init__()
        self.tok, self.texts = tok, texts
        self.cfg = types.SimpleNamespace(block_size=block_size, vocab_size=tok.get_vocab_size())
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        logits = torch.full((B, T, self.cfg.vocab_size), -1e4, device=idx.device)
        for i in range(B):
            s = self.tok.decode(idx[i, 1:].tolist())  # без ведущего EOS
            ref = next(t for t in self.texts if t.startswith(s) and (len(t) >= len(s)))
            nxt = 0 if len(s) >= len(ref) else self.tok.token_to_id(ref[len(s)])
            logits[i, -1, nxt] = 1e4
        return logits, None


def _setup_stub(rows):
    texts = [example_text(r) for r in rows]
    tok = CharTokenizer.train(texts)
    return tok, texts


def test_accuracy_oracle_100_and_wrong_0():
    rows = make_rows(30)
    tok, texts = _setup_stub(rows)
    res = evaluate_accuracy(OracleLM(tok, texts), tok, rows, batch_size=7)
    assert res["answer_accuracy"] == 1.0 and res["exact_solution"] == 1.0 and res["n"] == 30
    assert sum(v["n"] for v in res["by_op"].values()) == 30 and set(res["by_op"]) == {"+", "-"}
    assert sum(v["n"] for v in res["by_digits"].values()) == 30 and res["too_long"] == 0
    wrong = []
    for r, t in zip(rows, texts):
        a = r["answer"]
        wrong.append(t[: len(t) - len(a)] + str(int(a) + 1))
    # неверные ответы: тексты отличаются от эталона, поэтому нужен Oracle, выдающий wrong
    tok2 = _tok_with(wrong, tok)
    res0 = evaluate_accuracy(OracleLM(tok2, wrong), tok2, rows, batch_size=7)
    assert res0["answer_accuracy"] == 0.0 and res0["exact_solution"] == 0.0 and res0["no_eos"] == 0


def _tok_with(texts, base):
    return CharTokenizer.train(texts + ["".join(base.chars)])


def test_garbage_and_overlong_are_errors():
    rows = make_rows(6)
    tok, texts = _setup_stub(rows)
    # block_size мал: ни один пример не помещается -> все ошибки, генерации нет
    res = evaluate_accuracy(OracleLM(tok, texts, block_size=8), tok, rows)
    assert res["answer_accuracy"] == 0.0 and res["too_long"] == 6
    # мусор: модель всё время выдаёт '0' и никогда EOS -> ошибка
    class Junk(OracleLM):
        def forward(self, idx, targets=None):
            B, T = idx.shape
            lg = torch.full((B, T, self.cfg.vocab_size), -1e4)
            lg[..., self.tok.token_to_id("0")] = 1e4
            return lg, None
    res = evaluate_accuracy(Junk(tok, texts), tok, rows)
    assert res["answer_accuracy"] == 0.0 and res["no_eos"] == 6


def test_eval_keeps_train_mode_and_rng():
    rows = make_rows(12)
    tok, texts = _setup_stub(rows)
    mcfg = tiny_config().model
    mcfg.vocab_size, mcfg.block_size = tok.get_vocab_size(), 64
    model = build_model(mcfg)
    gen = make_batch_generator("cpu", 1)
    model.train()
    torch.manual_seed(0)
    before = (torch.get_rng_state().clone(), gen.get_state().clone())
    evaluate_accuracy(model, tok, rows, batch_size=5, max_new_tokens=5)
    assert model.training
    assert torch.equal(before[0], torch.get_rng_state()) and torch.equal(before[1], gen.get_state())
    model.eval()
    evaluate_accuracy(model, tok, rows, batch_size=5, max_new_tokens=3)
    assert not model.training
    assert all(p.grad is None for p in model.parameters())


# --- обучение с точностью -----------------------------------------------------------
def test_train_with_accuracy(alg_root, tmp_path):
    root, cfg = alg_root
    m = train(cfg, tmp_path, data_root=root, device="cpu", log=lambda *_: None)
    assert m["finished"] and [h["step"] for h in m["accuracy_history"]] == [3]
    assert m["accuracy_history"][0]["n"] == 6 and m["final_accuracy"]["n"] == 10
    assert 0.0 <= m["final_accuracy"]["answer_accuracy"] <= 1.0
    # те же веса, что без оценки точности: оценка не влияет на обучение (ГСЧ, режим)
    cfg2 = algebra_config(root, cfg.data.vocab_size, accuracy_interval=0, accuracy_final_samples=0)
    train(cfg2, tmp_path / "noacc", data_root=root, device="cpu", log=lambda *_: None)
    from llmlab.checkpoint import load_checkpoint
    a, b = load_checkpoint(tmp_path / "latest.pt"), load_checkpoint(tmp_path / "noacc" / "latest.pt")
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k


def test_generate_text_with_char_tokenizer(alg_root):
    from llmlab.generate import generate_text
    root, cfg = alg_root
    tok = load_corpus_tokenizer(corpus_dir(cfg))
    model = build_model(cfg.model)
    out = generate_text(model, tok, "1 + 2 = ", max_new_tokens=6, temperature=0.8, top_k=40, seed=1)
    assert isinstance(out, str) and out.startswith("1 + 2 = ")
    assert generate_text(model, tok, "1 + 2 = ", 4, 0.0, None).startswith("1 + 2 = ")


# --- регрессия ru_stories -------------------------------------------------------------
def test_ru_stories_config_unchanged():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "baseline.yaml")
    assert (cfg.data.dataset, cfg.data.tokenizer, cfg.data.vocab_size) == ("ru_stories", "bpe", 4000)
    assert corpus_dir(cfg).as_posix() == "data/vocab4000"
    assert cfg.train.accuracy_interval == 0 and cfg.model.vocab_size == 4000 and cfg.train.max_steps == 6000
    assert build_model(cfg.model).num_params() == 2_093_904
    for p in (root / "configs" / "variants").glob("*.yaml"):
        assert load_config(p).data.dataset == "ru_stories"
    alg = load_config(root / "configs" / "algebra" / "baseline.yaml")
    assert alg.data.dataset == "algebra_cot" and corpus_dir(alg).name == "algebra_cot_char"
    with pytest.raises(ValueError):
        Config.from_dict({"data": {"dataset": "nope"}})
