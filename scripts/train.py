"""Обучение с автопродолжением: python scripts/train.py -c configs/baseline.yaml --ckpt-dir checkpoints/baseline"""
import argparse

from llmlab.config import load_config
from llmlab.train import train

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", default="configs/baseline.yaml")
ap.add_argument("--ckpt-dir", default=None, help="по умолчанию checkpoints/<name>")
ap.add_argument("--set", nargs="*", default=[], help="переопределения key=value, напр. train.max_steps=20")
ap.add_argument("--no-resume", action="store_true")
ap.add_argument("--until-step", type=int, default=None, help="остановиться на шаге (имитация обрыва)")
a = ap.parse_args()
cfg = load_config(a.config, a.set)
m = train(cfg, a.ckpt_dir or f"checkpoints/{cfg.name}", resume=not a.no_resume, until_step=a.until_step)
print({k: m[k] for k in ("finished", "step", "n_params", "final_val_loss", "tokens_per_s_mean")})
