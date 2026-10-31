"""Параметры конфигов; код возврата 1, если отличие от базы больше ±5%.
python scripts/compare_params.py --base configs/baseline.yaml configs/variants/*.yaml"""
import argparse
import sys

from llmlab.params import compare, format_table

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="configs/baseline.yaml")
ap.add_argument("--tol", type=float, default=0.05)
ap.add_argument("configs", nargs="*")
a = ap.parse_args()
rows, ok = compare(a.base, a.configs, a.tol)
print(format_table(rows, a.tol))
if not ok:
    print(f"ОШИБКА: есть конфиги с отличием от базы больше ±{a.tol:.0%}", file=sys.stderr)
    sys.exit(1)
