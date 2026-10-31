"""Скачать данные, разбить, убрать дубли, обучить BPE и записать memmap: python scripts/prepare_data.py -c configs/baseline.yaml"""
import argparse

from llmlab.config import load_config
from llmlab.data import prepare

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", default="configs/baseline.yaml")
ap.add_argument("--set", nargs="*", default=[], help="переопределения key=value")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
prepare(load_config(a.config, a.set), force=a.force)
