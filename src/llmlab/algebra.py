"""Датасет ozlobin/simple_algebra_CoT_1m: разбор CSV, текст примера, дедупликация, сплиты.

Колонки: instruction ('341 - 745 = '), answer (-404), reasoning (многострочное решение, поле CSV в кавычках).
Лицензия в метаданных датасета не указана; данные в репозиторий не попадают (каталог data/ в .gitignore).

Текст обучающего примера: instruction + решение, затем EOS (EOS добавляет токенизация). В reasoning условие
встречается в трёх видах (проверено по всем 1 000 000 строк):
  A. reasoning начинается с самого instruction ('341 - 745 = -(745 - 341)\\n...')  -> берём reasoning как есть;
  B. первая строка - условие без '=' ('566 - 367\\n16 - 7 = 9...')                 -> строку-повтор убираем,
     после instruction ставим перевод строки;
  C. условия нет (первая строка - дополненные нулями операнды '702 + 089')         -> instruction + reasoning.
Во всех случаях подсказка = instruction дословно и является префиксом обучающего текста.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

_INSTR_RE = re.compile(r"^\s*(-?\d+)\s*([+-])\s*(-?\d+)\s*=\s*$")


def read_csv(path: str | Path) -> list[dict]:
    """Читает train.csv модулем csv (reasoning многострочное). Строки с пустым instruction/answer/reasoning -> ошибка."""
    csv.field_size_limit(1 << 30)
    out = []
    with open(path, encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            ex = {k: row.get(k) for k in ("instruction", "answer", "reasoning")}
            if any(v is None or v == "" for v in ex.values()):
                raise ValueError(f"{path}: строка {i + 2}: пустое поле {ex!r}")
            out.append(ex)
    return out


def parse_instruction(instruction: str) -> tuple[int, str, int]:
    m = _INSTR_RE.match(instruction)
    if not m:
        raise ValueError(f"не удалось разобрать instruction: {instruction!r}")
    return int(m.group(1)), m.group(2), int(m.group(3))


def example_prompt(ex: dict) -> str:
    """Подсказка при генерации: instruction дословно ('341 - 745 = ')."""
    return ex["instruction"]


def example_text(ex: dict) -> str:
    """Обучающий текст без EOS: instruction + решение без повтора условия (см. docstring модуля)."""
    ins, rea = ex["instruction"], ex["reasoning"]
    if rea.startswith(ins):  # A
        return rea
    base = ins.rstrip()
    if base.endswith("="):
        base = base[:-1].rstrip()
    first, nl, rest = rea.partition("\n")
    if first == base:  # B
        return ins + "\n" + rest
    return ins + rea  # C


def normalize_instruction(instruction: str) -> str:
    """Ключ дедупликации: без пробельных символов (регистра у цифр нет)."""
    return re.sub(r"\s+", "", instruction)


def dedup_and_split(rows: list[dict], val_size: int, test_size: int, seed: int):
    """Дедупликация по нормализованному instruction ДО разбиения (остаётся первое вхождение), затем
    фиксированный seed: val = первые val_size индексов перестановки, test = следующие test_size, остальное train.
    Порядок внутри train исходный. Возвращает (train, val, test, отчёт); пересечения проверяются."""
    seen: set[str] = set()
    uniq = []
    for r in rows:
        k = normalize_instruction(r["instruction"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    n = len(uniq)
    if val_size + test_size >= n:
        raise ValueError(f"val_size+test_size={val_size + test_size} не меньше числа уникальных примеров {n}")
    perm = np.random.default_rng(seed).permutation(n)
    val = [uniq[i] for i in perm[:val_size].tolist()]
    test = [uniq[i] for i in perm[val_size:val_size + test_size].tolist()]
    held = set(perm[:val_size + test_size].tolist())
    train = [r for i, r in enumerate(uniq) if i not in held]
    overlap = check_no_overlap(train, val, test)
    report = {
        "normalization": "удалены все пробельные символы в instruction; остаётся первое вхождение",
        "rows_total": len(rows), "duplicates_removed": len(rows) - n, "unique": n,
        "train": len(train), "val": len(val), "test": len(test),
        "split_seed": seed, "overlap_instruction": overlap,
    }
    return train, val, test, report


def check_no_overlap(train: list[dict], val: list[dict], test: list[dict]) -> dict:
    """Число общих нормализованных instruction между сплитами; при ненулевом значении - ошибка."""
    ks = [{normalize_instruction(r["instruction"]) for r in part} for part in (train, val, test)]
    res = {"train_val": len(ks[0] & ks[1]), "train_test": len(ks[0] & ks[2]), "val_test": len(ks[1] & ks[2])}
    if any(res.values()):
        raise AssertionError(f"instruction пересекаются между сплитами: {res}")
    return res


def composition(rows: list[dict]) -> dict:
    """Состав: операции, число разрядов большего по модулю операнда, диапазон операндов, знаки."""
    ops, digits, signs = Counter(), Counter(), Counter()
    lo, hi = None, None
    for r in rows:
        a, op, b = parse_instruction(r["instruction"])
        ops[op] += 1
        digits[max(len(str(abs(a))), len(str(abs(b))))] += 1
        signs[("-" if a < 0 else "+") + ("-" if b < 0 else "+")] += 1
        m, M = min(a, b), max(a, b)
        lo = m if lo is None else min(lo, m)
        hi = M if hi is None else max(hi, M)
    return {"ops": dict(ops), "max_operand_digits": dict(sorted(digits.items())), "operand_range": [lo, hi],
            "operand_signs": dict(signs)}


def write_examples(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_examples(path: Path, limit: int | None = None) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
            if limit is not None and len(out) >= limit:
                break
    return out
