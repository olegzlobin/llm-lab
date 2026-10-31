"""Логирование в W&B (режим из конфига/окружения) с заглушкой для disabled."""
from __future__ import annotations

import os
import uuid


class NullLogger:
    """Ничего не отправляет (режим disabled, тесты)."""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:8]

    def log(self, data: dict) -> None:
        pass

    def log_table(self, key: str, columns: list[str], rows: list[list]) -> None:
        pass

    def update_config(self, data: dict) -> None:
        pass

    def finish(self) -> None:
        pass


class WandbLogger:
    def __init__(self, cfg, run_id: str | None, mode: str):
        import wandb  # noqa: PLC0415  (импорт только если нужен)

        self.wandb = wandb
        self.run_id = run_id or wandb.util.generate_id()
        self.run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.name,
            id=self.run_id,
            resume="allow",
            config=cfg.to_dict(),
            mode=mode,
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    def log(self, data: dict) -> None:
        self.wandb.log(data)

    def log_table(self, key, columns, rows) -> None:
        self.wandb.log({key: self.wandb.Table(columns=columns, data=rows)})

    def update_config(self, data: dict) -> None:
        self.run.config.update(data, allow_val_change=True)

    def finish(self) -> None:
        self.wandb.finish()


def resolve_mode(cfg) -> str:
    return os.environ.get("WANDB_MODE") or cfg.wandb.mode


def make_logger(cfg, run_id: str | None):
    mode = resolve_mode(cfg)
    if mode == "disabled":
        return NullLogger(run_id)
    return WandbLogger(cfg, run_id, mode)
