"""Статистика корпуса (ru_stories: словари 2000/4000/8000; algebra_cot: реальный словарь, длины, block_size):
python scripts/measure_corpus.py -c configs/baseline.yaml"""
import argparse
import json

from llmlab.config import load_config
from llmlab.corpus_stats import format_algebra_markdown, format_markdown, measure

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", default="configs/baseline.yaml")
ap.add_argument("--vocab", nargs="*", type=int, default=[2000, 4000, 8000])
ap.add_argument("--json-out", default=None)
a = ap.parse_args()
cfg = load_config(a.config)
m = measure(cfg, a.vocab)  # для algebra_cot --vocab игнорируется: берётся реальный словарь
print(format_algebra_markdown(m) if cfg.data.dataset == "algebra_cot" else format_markdown(m))
print("\nОтчёт о дублях:\n" + json.dumps(m["dup_report"], ensure_ascii=False, indent=2))
if a.json_out:
    open(a.json_out, "w", encoding="utf-8").write(json.dumps(m, ensure_ascii=False, indent=2))
