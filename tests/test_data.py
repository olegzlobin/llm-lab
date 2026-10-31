import numpy as np
import torch

from conftest import make_stories
from llmlab.data import (dedup_splits, fixed_eval_offsets, make_splits, normalize, random_batch, story_text,
                         window_batch, make_batch_generator)


def test_story_text_format():
    s = {"storytitle": "Т", **{f"sentence{i}": f"П{i}." for i in range(1, 6)}}
    assert story_text(s) == "Т\nП1. П2. П3. П4. П5."


def test_splits_disjoint_and_dedup():
    train_raw = make_stories(400, 1, "tr")
    test_raw = make_stories(50, 2, "te")
    test_raw[0] = {**train_raw[3], "storyid": "te-dup"}  # дубль train (другой id)
    test_raw[1] = {**test_raw[2], "storyid": "te-self"}  # дубль внутри test
    train, val, test, rep = make_splits(train_raw, test_raw, 0.05, 1)
    assert len(val) == 20 and len(train) == 380
    n = lambda part: {normalize(story_text(s)) for s in part}  # noqa: E731
    assert not n(train) & n(val) and not n(train) & n(test) and not n(val) & n(test)
    ids = lambda part: {s["storyid"] for s in part}  # noqa: E731
    assert not ids(train) & ids(val)
    assert "te-dup" not in ids(test) and "te-2" not in ids(test) and "te-self" in ids(test)
    assert rep["test"]["removed_dup_of_train"] + rep["test"]["removed_dup_of_val"] >= 1
    assert rep["test"]["removed_internal_dup"] >= 1
    # детерминированность по seed
    again = make_splits(train_raw, test_raw, 0.05, 1)
    assert ids(again[1]) == ids(val)
    assert ids(make_splits(train_raw, test_raw, 0.05, 2)[1]) != ids(val)


def test_val_dup_of_train_removed():
    train = make_stories(30, 5, "a")
    val = [{**train[0], "storyid": "v0"}, {**make_stories(1, 9, "v")[0]}]
    v, t, rep = dedup_splits(train, val, [])
    assert rep["val"]["removed_dup_of_train"] == 1 and len(v) == 1


def test_batches_and_fixed_eval(corpus_root):
    data = torch.from_numpy(np.fromfile(corpus_root / "vocab300" / "train.bin", dtype=np.uint16).astype(np.int32))
    gen = make_batch_generator("cpu", 0)
    x, y = random_batch(data, 16, 4, gen)
    assert x.shape == (4, 16) and torch.equal(x[:, 1:], y[:, :-1])
    o1 = fixed_eval_offsets(data.numel(), 16, 4, 3, seed=5)
    o2 = fixed_eval_offsets(data.numel(), 16, 4, 3, seed=5)
    assert torch.equal(o1, o2) and not torch.equal(o1, fixed_eval_offsets(data.numel(), 16, 4, 3, seed=6))
    xa, _ = window_batch(data, o1[0], 16)
    xb, _ = window_batch(data, o2[0], 16)
    assert torch.equal(xa, xb)


def test_corpus_has_eos_and_uint16(corpus_root):
    import json
    meta = json.loads((corpus_root / "vocab300" / "meta.json").read_text())
    arr = np.fromfile(corpus_root / "vocab300" / "train.bin", dtype=np.uint16)
    assert arr.size == meta["splits"]["train"]["tokens"]
    assert (arr == meta["eos_id"]).sum() == meta["splits"]["train"]["stories"]
