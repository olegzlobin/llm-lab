"""Сравнение числа параметров конфигов с базовым."""
from __future__ import annotations

from .config import load_config
from .model import build_model


def count_params(config_path: str) -> int:
    return build_model(load_config(config_path).model).num_params()


def compare(base: str, others: list[str], tol: float = 0.05):
    """Возвращает (строки таблицы, ok). Строка: (путь, имя, params, отклонение от базы, в допуске)."""
    base_n = count_params(base)
    rows = [(base, load_config(base).name, base_n, 0.0, True)]
    ok = True
    for p in others:
        n = count_params(p)
        dev = n / base_n - 1.0
        good = abs(dev) <= tol
        ok &= good
        rows.append((p, load_config(p).name, n, dev, good))
    return rows, ok


def format_table(rows, tol: float) -> str:
    lines = [f"{'config':<44} {'name':<18} {'params':>12} {'vs base':>9}"]
    for path, name, n, dev, good in rows:
        lines.append(f"{path:<44} {name:<18} {n:>12,} {dev:>+8.2%} {'' if good else ' FAIL (> ±%.0f%%)' % (tol * 100)}")
    return "\n".join(lines)
