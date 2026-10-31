"""Точность решения на val или test (algebra_cot): python scripts/evaluate_accuracy.py --ckpt checkpoints/algebra_baseline/latest.pt --split val --n 2000"""
import argparse

import torch

from llmlab.algebra import read_examples
from llmlab.checkpoint import load_checkpoint
from llmlab.config import Config
from llmlab.data import corpus_dir, load_corpus_tokenizer
from llmlab.metrics import evaluate_accuracy, format_table
from llmlab.model import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--split", choices=["val", "test"], default="val")
ap.add_argument("--n", type=int, default=2000, help="первые N примеров сплита")
ap.add_argument("--batch-size", type=int, default=256)
ap.add_argument("--data-dir", default=None, help="корень данных (по умолчанию из конфига чекпойнта)")
ap.add_argument("--device", default=None)
a = ap.parse_args()
ck = load_checkpoint(a.ckpt)
cfg = Config.from_dict(ck["config"])
if cfg.data.dataset != "algebra_cot":
    raise SystemExit("метрика точности есть только для data.dataset=algebra_cot")
dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
model = build_model(cfg.model)
model.load_state_dict(ck["model"])
model.to(dev)
cdir = corpus_dir(cfg, a.data_dir)
examples = read_examples(cdir / f"{a.split}.jsonl", a.n)
print(f"{cfg.name}, шаг {ck['step']}, сплит {a.split}")
print(format_table(evaluate_accuracy(model, load_corpus_tokenizer(cdir), examples, a.batch_size)))
