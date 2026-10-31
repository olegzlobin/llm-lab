"""Атомарная запись и загрузка чекпойнтов, RNG-состояния."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

CKPT_NAME = "latest.pt"


def save_atomic(path: str | os.PathLike, state: dict) -> None:
    """Временный файл в том же каталоге + fsync + os.replace: читатель видит старый или новый файл целиком."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def load_checkpoint(path: str | os.PathLike) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def get_rng_state(batch_gen: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "batch_gen": batch_gen.get_state(),
    }


def set_rng_state(state: dict, batch_gen: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    batch_gen.set_state(state["batch_gen"])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
