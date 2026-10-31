"""Обучение: AdamW, warmup+cosine, clipping, AMP, torch.compile, атомарные чекпойнты и resume."""
from __future__ import annotations

import json
import math
import os
import signal
import threading
import time
from pathlib import Path

import torch
import torch.nn as nn

from .checkpoint import CKPT_NAME, get_rng_state, load_checkpoint, save_atomic, set_rng_state, set_seed
from .config import Config
from .data import (corpus_dir, fixed_eval_offsets, load_corpus_tokenizer, load_tokens_tensor,
                   make_batch_generator, random_batch, window_batch)
from .generate import generate_text
from .model import build_model
from .tracking import make_logger, resolve_mode


def lr_at(step: int, lr: float, min_lr_ratio: float, warmup: int, max_steps: int) -> float:
    """Линейный warmup, затем cosine до lr*min_lr_ratio. step - число уже сделанных шагов."""
    if warmup > 0 and step < warmup:
        return lr * (step + 1) / warmup
    if step >= max_steps:
        return lr * min_lr_ratio
    progress = (step - warmup) / max(1, max_steps - warmup)
    coef = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr * (min_lr_ratio + (1.0 - min_lr_ratio) * coef)


def make_optimizer(model: nn.Module, tcfg, device_type: str) -> torch.optim.Optimizer:
    """Weight decay только для матриц весов; без norm, bias и эмбеддингов (в т.ч. tied lm_head)."""
    emb = {id(p) for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()}
    decay, no_decay = [], []
    for p in model.parameters():  # parameters() не дублирует tied-веса
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 and id(p) not in emb else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": tcfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tcfg.lr, betas=tuple(tcfg.betas), fused=(device_type == "cuda"))


def resolve_runtime(tcfg, device: torch.device) -> dict:
    """Какие ускорения реально включены. Записывается в метаданные запуска и чекпойнт."""
    cuda = device.type == "cuda"
    amp = str(tcfg.amp_dtype).lower()
    if not cuda or amp == "none" or amp == "false":
        amp = "none"
    elif amp == "auto":
        amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if amp not in ("none", "fp16", "bf16"):
        raise ValueError(f"amp_dtype={tcfg.amp_dtype!r}: ожидается auto|fp16|bf16|none")
    comp = str(tcfg.compile).lower()
    compile_on = cuda if comp == "auto" else comp in ("true", "1", "yes")
    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if cuda else None,
        "amp_dtype": amp,
        "grad_scaler": amp == "fp16",
        "compile": compile_on,
        "compile_backend": os.environ.get("LLMLAB_COMPILE_BACKEND", "inductor") if compile_on else None,
        "fused_adamw": cuda,
        "matmul_precision": "high",
        "data_on_device": True,
        "torch": torch.__version__,
    }


@torch.no_grad()
def evaluate(model, data, offsets, block_size, amp_ctx) -> float:
    """Eager-модель, фиксированные окна. Вызывающий восстанавливает train()."""
    model.eval()
    total = torch.zeros((), device=data.device)
    for offs in offsets:
        x, y = window_batch(data, offs.to(data.device), block_size)
        with amp_ctx():
            _, loss = model(x, y)
        total += loss.float()
    return (total / len(offsets)).item()


def train(cfg: Config, ckpt_dir: str | os.PathLike, data_root: str | os.PathLike | None = None,
          resume: bool = True, until_step: int | None = None, device: str | None = None, log=print) -> dict:
    """Обучение с автопродолжением из ckpt_dir/latest.pt. until_step - остановиться раньше max_steps
    (имитация обрыва, чекпойнт при этом пишется)."""
    tc, mc = cfg.train, cfg.model
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rt = resolve_runtime(tc, dev)
    torch.set_float32_matmul_precision("high")

    cdir = corpus_dir(cfg, data_root)
    train_data = load_tokens_tensor(cdir, "train", dev)
    val_data = load_tokens_tensor(cdir, "val", dev)
    tok = load_corpus_tokenizer(cdir)

    set_seed(cfg.seed)
    model = build_model(mc).to(dev)
    n_params = model.num_params()
    opt = make_optimizer(model, tc, dev.type)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(rt["amp_dtype"])
    scaler = torch.amp.GradScaler("cuda", enabled=rt["grad_scaler"])
    batch_gen = make_batch_generator(dev, cfg.seed)

    def amp_ctx():
        return torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_dtype is not None)

    step, history, elapsed_prev, run_id = 0, [], 0.0, None
    ckpt_path = ckpt_dir / CKPT_NAME
    if resume and ckpt_path.exists():
        ck = load_checkpoint(ckpt_path)
        if ck["config"]["model"] != cfg.to_dict()["model"]:
            raise ValueError("Конфиг модели в чекпойнте отличается от текущего; используйте другой ckpt-dir")
        model.load_state_dict(ck["model"])  # state_dict исходного (не compiled) модуля
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"], batch_gen)
        step, history, elapsed_prev, run_id = ck["step"], ck["history"], ck["elapsed"], ck["wandb_run_id"]
        log(f"Продолжаю с шага {step} из {ckpt_path}")

    train_model = model
    if rt["compile"]:  # компилируется только train-путь; eval и генерация идут на eager-модуле
        train_model = torch.compile(model, backend=rt["compile_backend"])

    logger = make_logger(cfg, run_id)
    logger.update_config({"n_params": n_params, "runtime": rt})
    log(f"Параметры: {n_params:,}; runtime: {rt}")

    eval_offsets = fixed_eval_offsets(val_data.numel(), mc.block_size, tc.batch_size, tc.eval_batches, tc.eval_seed)
    tokens_per_step = tc.batch_size * tc.grad_accum * mc.block_size
    target = tc.max_steps if until_step is None else min(until_step, tc.max_steps)

    state = {"stop": False}
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for s in (signal.SIGINT, signal.SIGTERM):
            old_handlers[s] = signal.signal(s, lambda *_: state.__setitem__("stop", True))

    session_start = time.time()
    last_ckpt_time = time.time()
    samples: list = []
    tps_log: list[float] = []

    def elapsed() -> float:
        return elapsed_prev + time.time() - session_start

    def save(final: bool = False) -> None:
        nonlocal last_ckpt_time
        save_atomic(ckpt_path, {
            "model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
            "step": step, "rng": get_rng_state(batch_gen), "config": cfg.to_dict(), "wandb_run_id": logger.run_id,
            "history": history, "elapsed": elapsed(), "runtime": rt,
        })
        last_ckpt_time = time.time()

    def do_samples() -> None:
        rows = []
        for i, pr in enumerate(tc.sample_prompts):
            text = generate_text(model, tok, pr, tc.sample_max_new_tokens, tc.sample_temperature,
                                 tc.sample_top_k, seed=tc.eval_seed + i, device=dev)
            rows.append([step, pr, text])
        model.train()
        samples[:] = rows
        logger.log_table("samples", ["step", "prompt", "text"], rows)

    def do_eval(train_loss: float | None) -> None:
        v = evaluate(model, val_data, eval_offsets, mc.block_size, amp_ctx)
        model.train()
        if not math.isfinite(v):
            raise FloatingPointError(f"val loss не конечен ({v}) на шаге {step}")
        rec = {"step": step, "train_loss": train_loss, "val_loss": v, "tokens_seen": step * tokens_per_step}
        history.append(rec)
        logger.log({"step": step, "val/loss": v})
        log(f"step {step}: val loss {v:.4f}")

    model.train()
    acc = torch.zeros((), device=dev)
    acc_n, last_train, t0, steps_since = 0, None, time.time(), 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        while step < target and not state["stop"]:
            lr = lr_at(step, tc.lr, tc.min_lr_ratio, tc.warmup_steps, tc.max_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            for _ in range(tc.grad_accum):
                x, y = random_batch(train_data, mc.block_size, tc.batch_size, batch_gen)
                with amp_ctx():
                    _, loss = train_model(x, y)
                scaler.scale(loss / tc.grad_accum).backward()
                acc += loss.detach().float() / tc.grad_accum
            acc_n += 1
            if tc.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1
            steps_since += 1

            if step % tc.log_interval == 0 or step == target:  # единственная синхронизация с хоста
                last_train = (acc / acc_n).item()
                if not math.isfinite(last_train):
                    raise FloatingPointError(
                        f"train loss не конечен ({last_train}) к шагу {step}: переполнение/NaN. "
                        f"Уменьшите lr, увеличьте warmup_steps или поставьте amp_dtype: none/bf16.")
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                tps = steps_since * tokens_per_step / (time.time() - t0)
                tps_log.append(tps)
                rec = {"step": step, "train/loss": last_train, "lr": lr, "tokens_per_s": tps}
                if dev.type == "cuda":
                    rec["gpu/peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
                logger.log(rec)
                log(f"step {step}: loss {last_train:.4f} lr {lr:.2e} {tps:,.0f} tok/s")
                acc.zero_(); acc_n = 0

            if step % tc.eval_interval == 0 or step == tc.max_steps:
                do_eval(last_train)
            if step % tc.sample_interval == 0 or step == tc.max_steps:
                do_samples()
            if step % tc.ckpt_interval == 0 and time.time() - last_ckpt_time >= tc.ckpt_min_seconds:
                save()
            if step % tc.log_interval == 0 or step % tc.eval_interval == 0:
                t0, steps_since = time.time(), 0
        if step == tc.max_steps and (not history or history[-1]["step"] != step):
            do_eval(last_train)
            do_samples()
        save(final=True)
    finally:
        for s, h in old_handlers.items():
            signal.signal(s, h)

    finished = step >= tc.max_steps
    metrics = {
        "name": cfg.name, "run_id": logger.run_id, "finished": finished, "step": step,
        "n_params": n_params, "arch": mc.arch, "tokens_seen": step * tokens_per_step,
        "final_train_loss": last_train if last_train is not None else (history[-1]["train_loss"] if history else None),
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "best_val_loss": min((h["val_loss"] for h in history), default=None),
        "tokens_per_s_mean": sum(tps_log) / len(tps_log) if tps_log else None,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None,
        "train_seconds": elapsed(), "runtime": rt, "history": history, "samples": samples, "config": cfg.to_dict(),
        "wandb_mode": resolve_mode(cfg),
    }
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.finish()
    return metrics
