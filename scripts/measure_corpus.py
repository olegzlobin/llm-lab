"""Статистика корпуса для словарей 2000/4000/8000: python scripts/measure_corpus.py"""
import argparse
import json

from llmlab.config import load_config
from llmlab.corpus_stats import format_markdown, measure

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", default="configs/baseline.yaml")
ap.add_argument("--vocab", nargs="*", type=int, default=[2000, 4000, 8000])
ap.add_argument("--json-out", default=None)
a = ap.parse_args()
m = measure(load_config(a.config), a.vocab)
print(format_markdown(m))
print("\nОтчёт о дублях:\n" + json.dumps(m["dup_report"], ensure_ascii=False, indent=2))
if a.json_out:
    open(a.json_out, "w", encoding="utf-8").write(json.dumps(m, ensure_ascii=False, indent=2))
