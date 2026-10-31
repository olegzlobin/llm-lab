"""Статистика корпуса по сплитам для разных размеров словаря."""
from __future__ import annotations

import numpy as np

from pathlib import Path

from . import algebra
from .config import Config
from .data import download_file, get_splits, story_text
from .tokenizer import CharTokenizer, encode_texts, train_char_tokenizer, train_tokenizer

MB = 1024 * 1024


def measure_algebra(cfg: Config) -> dict:
    """algebra_cot: состав операций и диапазоны, токены по сплитам (реальный словарь), длины, block_size."""
    d = cfg.data
    raw = Path(d.data_dir) / "raw" / "algebra" / d.train_file
    if not raw.exists():
        download_file(d.repo_id, d.train_file, raw)
    rows = algebra.read_csv(raw)
    train, val, test, report = algebra.dedup_and_split(rows, d.val_size, d.test_size, d.split_seed)
    texts = {k: [algebra.example_text(r) for r in v] for k, v in (("train", train), ("val", val), ("test", test))}
    tok = train_char_tokenizer(texts["train"]) if d.tokenizer == "char" else train_tokenizer(texts["train"], d.vocab_size)
    out = {"dup_report": report, "composition_raw": algebra.composition(rows), "vocab_size": tok.get_vocab_size(),
           "tokenizer": d.tokenizer, "splits": {}}
    all_lens = {}
    for k, tx in texts.items():
        if isinstance(tok, CharTokenizer):
            lens = np.array([len(t) + 1 for t in tx])  # +1 EOS (проверка: все символы из train; иначе кодирование упадёт)
            if k != "train":
                tok.encode_stream(tx)
        else:
            lens = np.array([len(i) + 1 for i in encode_texts(tok, tx)])
        all_lens[k] = lens
        out["splits"][k] = {"examples": len(tx), "tokens": int(lens.sum()), "mean_len": float(lens.mean()),
                            "p95_len": float(np.percentile(lens, 95)), "p99_len": float(np.percentile(lens, 99)),
                            "max_len": int(lens.max())}
        if k == "train":
            p99 = float(np.percentile(lens, 99))
            out["recommended_block_size"] = int(-(-int(np.ceil(p99)) // 8) * 8)
    bs = out["recommended_block_size"]
    # пример (EOS + токены текста; столько же контекста нужно при предсказании последнего EOS) должен быть <= block_size
    out["longer_than_block"] = {k: int((l > bs).sum()) for k, l in all_lens.items()}
    return out


def format_algebra_markdown(m: dict) -> str:
    L = [f"Токенизатор: {m['tokenizer']}, реальный размер словаря: {m['vocab_size']}", "",
         "| split | примеров | токенов (с EOS) | mean | p95 | p99 | max |", "|---|---|---|---|---|---|---|"]
    for k, r in m["splits"].items():
        L.append(f"| {k} | {r['examples']:,} | {r['tokens']:,} | {r['mean_len']:.1f} | {r['p95_len']:.0f} | "
                 f"{r['p99_len']:.0f} | {r['max_len']} |")
    L.append("")
    L.append(f"Рекомендуемый block_size (кратно 8, покрывает p99 train): {m['recommended_block_size']}")
    if m.get("longer_than_block"):
        L.append("Примеров длиннее block_size по сплитам: " + ", ".join(f"{k} {v}" for k, v in m["longer_than_block"].items()))
    c = m["composition_raw"]
    L.append("")
    L.append(f"Состав всего файла: операции {c['ops']}; разрядов у большего операнда {c['max_operand_digits']}; "
             f"диапазон операндов {c['operand_range']}; знаки операндов {c['operand_signs']}")
    return "\n".join(L)


def measure(cfg: Config, vocab_sizes=(2000, 4000, 8000)) -> dict:
    if cfg.data.dataset == "algebra_cot":
        return measure_algebra(cfg)
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
