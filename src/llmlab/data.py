"""Данные: загрузка ru_stories, сплиты, дедупликация, memmap и батчи.

Датасет inkoziev/ru_stories, лицензия CC-BY-NC-4.0. В git данные не попадают,
скачиваются в data_dir при запуске.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import torch

from .config import Config
from .tokenizer import EOS_TOKEN, encode_texts, eos_id, load_tokenizer, train_tokenizer

SENTENCE_KEYS = [f"sentence{i}" for i in range(1, 6)]


# --- загрузка -----------------------------------------------------------------
def download_file(repo_id: str, filename: str, dest: Path) -> Path:
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"Скачиваю {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(tmp, dest)
    return dest


def load_raw(cfg: Config, download: bool = True) -> tuple[list[dict], list[dict]]:
    raw = Path(cfg.data.data_dir) / "raw"
    out = []
    for fn in (cfg.data.train_file, cfg.data.test_file):
        p = raw / fn
        if not p.exists():
            if not download:
                raise FileNotFoundError(p)
            download_file(cfg.data.repo_id, fn, p)
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out[0], out[1]


# --- тексты, сплиты, дедупликация -----------------------------------------------
def story_text(story: dict) -> str:
    """Заголовок, перевод строки, затем пять предложений через пробел."""
    sents = " ".join(str(story.get(k) or "").strip() for k in SENTENCE_KEYS)
    return f"{str(story.get('storytitle') or '').strip()}\n{sents}"


def normalize(text: str) -> str:
    """Нормализация для поиска дублей: регистр, ё->е, пробелы."""
    return re.sub(r"\s+", " ", text.casefold().replace("ё", "е")).strip()


def split_train_val(stories: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Валидация выделяется из train по историям, фиксированный seed."""
    n = len(stories)
    n_val = int(round(n * val_fraction))
    perm = np.random.default_rng(seed).permutation(n)
    val_idx = set(perm[:n_val].tolist())
    train = [s for i, s in enumerate(stories) if i not in val_idx]
    val = [s for i, s in enumerate(stories) if i in val_idx]
    return train, val


def dedup_splits(train: list[dict], val: list[dict], test: list[dict]):
    """Убирает из val и test истории, чей нормализованный текст есть в train/val или уже встречался
    в том же сплите. train не меняется. Возвращает (val, test, отчёт)."""
    norm = lambda s: normalize(story_text(s))  # noqa: E731
    train_set = {norm(s) for s in train}
    train_seen: set[str] = set()
    train_internal = 0
    for s in train:
        n = norm(s)
        train_internal += n in train_seen
        train_seen.add(n)

    examples: dict[str, list[str]] = {"val_vs_train": [], "test_vs_train": [], "test_vs_val": []}
    val_orig_set = {norm(s) for s in val}

    new_val, seen = [], set()
    rm_val_train = rm_val_int = 0
    for s in val:
        n = norm(s)
        if n in train_set:
            rm_val_train += 1
            examples["val_vs_train"].append(str(s.get("storyid")))
        elif n in seen:
            rm_val_int += 1
        else:
            seen.add(n)
            new_val.append(s)

    new_test, seen = [], set()
    rm_test_train = rm_test_val = rm_test_int = 0
    for s in test:
        n = norm(s)
        if n in train_set:
            rm_test_train += 1
            examples["test_vs_train"].append(str(s.get("storyid")))
        elif n in val_orig_set:
            rm_test_val += 1
            examples["test_vs_val"].append(str(s.get("storyid")))
        elif n in seen:
            rm_test_int += 1
        else:
            seen.add(n)
            new_test.append(s)

    train_ids = {str(s.get("storyid")) for s in train}
    report = {
        "normalization": "casefold, ё->е, схлопывание пробелов; сравнивается текст 'заголовок + предложения'",
        "train": {"stories": len(train), "internal_duplicates_kept": train_internal},
        "val": {
            "before": len(val),
            "removed_dup_of_train": rm_val_train,
            "removed_internal_dup": rm_val_int,
            "after": len(new_val),
        },
        "test": {
            "before": len(test),
            "removed_dup_of_train": rm_test_train,
            "removed_dup_of_val": rm_test_val,
            "removed_internal_dup": rm_test_int,
            "after": len(new_test),
        },
        "storyid_overlap": {
            "test_with_train": len({str(s.get("storyid")) for s in test} & train_ids),
        },
        "examples_storyid": {k: v[:5] for k, v in examples.items()},
    }
    return new_val, new_test, report


def get_splits(cfg: Config, download: bool = True):
    """train/val/test после дедупликации + отчёт. Детерминированно по split_seed."""
    raw_train, raw_test = load_raw(cfg, download)
    return make_splits(raw_train, raw_test, cfg.data.val_fraction, cfg.data.split_seed)


def make_splits(raw_train, raw_test, val_fraction, split_seed):
    train, val = split_train_val(raw_train, val_fraction, split_seed)
    val, test, report = dedup_splits(train, val, raw_test)
    return train, val, test, report


# --- корпус (токенизатор + memmap) ----------------------------------------------
def corpus_dir(cfg: Config, root: str | os.PathLike | None = None) -> Path:
    return Path(root or cfg.data.data_dir) / f"vocab{cfg.data.vocab_size}"


def write_bin(path: Path, token_lists: list[list[int]], eos: int, vocab_size: int) -> int:
    """Истории подряд через EOS в uint16."""
    assert vocab_size <= 65536, "uint16 не вмещает такой словарь"
    parts = []
    for ids in token_lists:
        parts.append(np.asarray(ids, dtype=np.uint16))
        parts.append(np.asarray([eos], dtype=np.uint16))
    arr = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint16)
    arr.tofile(path)
    return int(arr.size)


def build_corpus(train, val, test, vocab_size: int, out_dir: Path) -> dict:
    """Токенизатор обучается только на train. Пишет tokenizer.json, {train,val,test}.bin, meta.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = {k: [story_text(s) for s in v] for k, v in (("train", train), ("val", val), ("test", test))}
    tok = train_tokenizer(texts["train"], vocab_size)
    tok.save(str(out_dir / "tokenizer.json"))
    eos = eos_id(tok)
    meta = {"vocab_size": vocab_size, "actual_vocab_size": tok.get_vocab_size(), "eos_id": eos, "splits": {}}
    for split, tx in texts.items():
        n = write_bin(out_dir / f"{split}.bin", encode_texts(tok, tx), eos, vocab_size)
        meta["splits"][split] = {"stories": len(tx), "tokens": n}
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def prepare(cfg: Config, download: bool = True, force: bool = False) -> dict:
    """Полная подготовка: скачать, разбить, убрать дубли, обучить BPE, записать memmap."""
    out = corpus_dir(cfg)
    if (out / "meta.json").exists() and not force:
        print(f"Корпус уже подготовлен: {out}")
        return json.loads((out / "meta.json").read_text(encoding="utf-8"))
    train, val, test, report = get_splits(cfg, download)
    splits_dir = Path(cfg.data.data_dir) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        (splits_dir / f"{name}_ids.json").write_text(json.dumps([s.get("storyid") for s in part]))
    (splits_dir / "dup_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Отчёт о дублях:\n" + json.dumps(report, ensure_ascii=False, indent=2))
    meta = build_corpus(train, val, test, cfg.data.vocab_size, out)
    meta["dup_report"] = report
    print(json.dumps({k: v for k, v in meta.items() if k != "dup_report"}, indent=2))
    return meta


def load_tokens(cdir: Path, split: str) -> np.memmap:
    return np.memmap(cdir / f"{split}.bin", dtype=np.uint16, mode="r")


def load_corpus_tokenizer(cdir: Path):
    return load_tokenizer(cdir / "tokenizer.json")


# --- батчи ---------------------------------------------------------------------
def load_tokens_tensor(cdir: Path, split: str, device) -> torch.Tensor:
    """Весь сплит целиком в памяти устройства (int32; корпус мал). Без DataLoader и копий на шаг."""
    arr = np.asarray(load_tokens(cdir, split), dtype=np.int32)
    return torch.from_numpy(arr).to(device)


def _check_len(n_tokens: int, block_size: int) -> int:
    n = n_tokens - block_size  # стартовые позиции 0..n-1 (нужен ещё +1 токен для цели)
    if n < 1:
        raise ValueError(f"Слишком мало токенов ({n_tokens}) для block_size={block_size}")
    return n


def make_batch_generator(device, seed: int) -> torch.Generator:
    """Генератор на том же устройстве, что и корпус; состояние сохраняется в чекпойнте."""
    return torch.Generator(device=device).manual_seed(seed)


def random_batch(data: torch.Tensor, block_size: int, batch_size: int, gen: torch.Generator):
    """Случайные окна индексацией прямо на устройстве."""
    n = _check_len(data.numel(), block_size)
    offs = torch.randint(0, n, (batch_size,), generator=gen, device=data.device)
    return window_batch(data, offs, block_size)


def window_batch(data: torch.Tensor, offsets: torch.Tensor, block_size: int):
    idx = offsets[:, None] + torch.arange(block_size + 1, device=data.device)[None, :]
    w = data[idx].long()
    return w[:, :-1], w[:, 1:]


def fixed_eval_offsets(n_tokens: int, block_size: int, batch_size: int, n_batches: int, seed: int) -> torch.Tensor:
    """Одни и те же окна для всех запусков с одним eval-seed (CPU-генератор, не зависит от устройства)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, _check_len(n_tokens, block_size), (n_batches, batch_size), generator=gen)
