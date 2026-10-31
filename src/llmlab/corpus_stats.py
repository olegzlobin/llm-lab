"""Статистика корпуса по сплитам для разных размеров словаря."""
from __future__ import annotations

import numpy as np

from .config import Config
from .data import get_splits, story_text
from .tokenizer import encode_texts, train_tokenizer

MB = 1024 * 1024


def measure(cfg: Config, vocab_sizes=(2000, 4000, 8000)) -> dict:
    train, val, test, report = get_splits(cfg)
    texts = {k: [story_text(s) for s in v] for k, v in (("train", train), ("val", val), ("test", test))}
    chars = {k: sum(len(t) for t in v) for k, v in texts.items()}
    nbytes = {k: sum(len(t.encode("utf-8")) for t in v) for k, v in texts.items()}
    out = {"dup_report": report, "text_mb": {k: nbytes[k] / MB for k in nbytes}, "chars": chars, "by_vocab": {}}
    out["text_mb"]["total"] = sum(nbytes.values()) / MB
    for V in vocab_sizes:
        tok = train_tokenizer(texts["train"], V)  # только train
        res = {}
        for k, tx in texts.items():
            lens = np.array([len(i) + 1 for i in encode_texts(tok, tx)])  # +1 EOS
            res[k] = {"stories": len(tx), "tokens": int(lens.sum()), "mean_len": float(lens.mean()),
                      "p95_len": float(np.percentile(lens, 95)), "chars_per_token": chars[k] / int(lens.sum()),
                      "bin_mb": int(lens.sum()) * 2 / MB}
        out["by_vocab"][V] = res
    return out


def format_markdown(m: dict) -> str:
    L = ["| vocab | split | stories | tokens (c EOS) | mean len | p95 len | chars/token | uint16 MB |",
         "|---|---|---|---|---|---|---|---|"]
    for V, res in m["by_vocab"].items():
        for k, r in res.items():
            L.append(f"| {V} | {k} | {r['stories']} | {r['tokens']:,} | {r['mean_len']:.1f} | {r['p95_len']:.0f} | "
                     f"{r['chars_per_token']:.2f} | {r['bin_mb']:.1f} |")
    mb = m["text_mb"]
    L.append("")
    L.append("Текст (UTF-8), МБ: " + ", ".join(f"{k} {v:.2f}" for k, v in mb.items()))
    return "\n".join(L)
