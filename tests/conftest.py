import random

import pytest

from llmlab.config import Config
from llmlab.data import build_corpus, make_splits

WORDS = ("кот пёс лес дом река мальчик девочка весна утро зима снег небо птица ветер дорога "
         "друг сад книга окно свет ночь море гора город поле").split()
VERBS = "увидел нашёл забыл любил искал принёс спросил услышал потерял".split()


def make_stories(n: int, seed: int, prefix: str) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        s = {"storyid": f"{prefix}-{i}", "storytitle": f"{rng.choice(WORDS).capitalize()} и {rng.choice(WORDS)}"}
        for k in range(1, 6):
            s[f"sentence{k}"] = f"{rng.choice(WORDS).capitalize()} {rng.choice(VERBS)} {rng.choice(WORDS)} {rng.choice(WORDS)}."
        out.append(s)
    return out


def tiny_config(**model) -> Config:
    d = {
        "name": "tiny", "seed": 7,
        "data": {"vocab_size": 300, "val_fraction": 0.05, "split_seed": 1},
        "model": {"block_size": 32, "n_layer": 2, "n_head": 2, "n_embd": 32, "dropout": 0.1, **model},
        "train": {"batch_size": 8, "max_steps": 10, "lr": 3e-3, "warmup_steps": 2, "eval_interval": 5,
                  "eval_batches": 3, "log_interval": 2, "ckpt_interval": 5, "sample_interval": 10,
                  "sample_max_new_tokens": 8, "sample_prompts": ["Кот\n"], "amp_dtype": "none", "compile": "false"},
        "wandb": {"mode": "disabled"},
    }
    return Config.from_dict(d)


@pytest.fixture(autouse=True)
def _no_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture(scope="session")
def corpus_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    train_raw = make_stories(500, 1, "tr")
    test_raw = make_stories(60, 2, "te")
    train, val, test, _ = make_splits(train_raw, test_raw, 0.05, 1)
    build_corpus(train, val, test, 300, root / "vocab300")
    return root
