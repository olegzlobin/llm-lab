"""Конфигурация эксперимента: датаклассы + загрузка из YAML с dotted-переопределениями."""
from __future__ import annotations

import copy
import dataclasses
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class DataConfig:
    repo_id: str = "inkoziev/ru_stories"
    train_file: str = "rustories.train.final2.json"
    test_file: str = "rustories.test.final2.json"
    data_dir: str = "data"
    vocab_size: int = 4000
    val_fraction: float = 0.05
    split_seed: int = 1234
    dataset: str = "ru_stories"  # ru_stories | algebra_cot
    tokenizer: str = "bpe"  # bpe | char (char: vocab_size = 1 + число символов train, проверяется в prepare_data)
    val_size: int = 20000  # algebra_cot: фиксированное число примеров в val
    test_size: int = 20000  # algebra_cot: фиксированное число примеров в test

    def __post_init__(self) -> None:
        if self.dataset not in ("ru_stories", "algebra_cot"):
            raise ValueError(f"data.dataset={self.dataset!r}: ожидается ru_stories | algebra_cot")
        if self.tokenizer not in ("bpe", "char"):
            raise ValueError(f"data.tokenizer={self.tokenizer!r}: ожидается bpe | char")


@dataclass
class ModelConfig:
    arch: str = "gpt2"
    vocab_size: int = 0  # 0 = взять из data.vocab_size
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 144
    dropout: float = 0.1
    bias: bool = False  # bias в Linear и LayerNorm (быстрее и проще без него)
    # заменяемые компоненты (arch: gpt2 собирается из них; значения по умолчанию = классический GPT-2)
    norm: str = "layernorm"  # layernorm | rmsnorm
    ffn: str = "gelu"  # gelu | swiglu
    positions: str = "learned"  # learned | rope
    attention: str = "full"  # full | gqa | sliding
    n_kv_heads: int | None = None  # для gqa (и опционально sliding); None = n_head
    window_size: int | None = None  # для sliding
    ffn_mult: float | None = None  # hidden = ffn_mult * n_embd (кратно 8); None: 4 для gelu, 8/3 для swiglu
    extra: dict = field(default_factory=dict)  # параметры, специфичные для будущих архитектур


@dataclass
class TrainConfig:
    batch_size: int = 64
    grad_accum: int = 1
    max_steps: int = 6000
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    amp_dtype: str = "auto"  # auto | fp16 | bf16 | none; только CUDA (auto: bf16 если поддерживается, иначе fp16)
    compile: str = "auto"  # auto (= включено на CUDA) | true | false
    ckpt_min_seconds: float = 0.0  # не писать плановый чекпойнт чаще (итоговый пишется всегда)
    eval_interval: int = 250
    eval_batches: int = 25
    eval_seed: int = 4242
    log_interval: int = 50
    ckpt_interval: int = 500
    sample_interval: int = 1000
    sample_max_new_tokens: int = 100
    sample_temperature: float = 0.8
    sample_top_k: int = 40
    accuracy_interval: int = 0  # algebra_cot: каждые N шагов точность на val[:accuracy_samples]; 0 = выключено
    accuracy_samples: int = 512
    accuracy_final_samples: int = 2000  # финальная оценка на val[:N] после обучения
    accuracy_batch_size: int = 256
    sample_prompts: list = field(default_factory=lambda: ["Новый год\n", "Старый пёс\nНа окраине деревни"])


@dataclass
class WandbConfig:
    mode: str = "online"  # online | offline | disabled; env WANDB_MODE имеет приоритет
    project: str = "llm-lab"
    entity: str | None = None


@dataclass
class Config:
    name: str = "baseline"
    seed: int = 1337
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        if self.model.vocab_size in (0, None):
            self.model.vocab_size = self.data.vocab_size
        elif self.model.vocab_size != self.data.vocab_size:
            raise ValueError(
                f"model.vocab_size={self.model.vocab_size} не совпадает с data.vocab_size={self.data.vocab_size}"
            )
        self.train.betas = tuple(self.train.betas)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return _build(cls, d)


def _build(cls, d: dict):
    """Строгая сборка датакласса: неизвестные ключи -> ошибка."""
    d = dict(d or {})
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(d) - set(fields)
    if unknown:
        raise ValueError(f"{cls.__name__}: неизвестные ключи {sorted(unknown)}")
    kwargs = {}
    for k, v in d.items():
        sub = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig, "wandb": WandbConfig}.get(k)
        kwargs[k] = _build(sub, v) if (sub is not None and cls is Config) else v
    return cls(**kwargs)


def apply_overrides(d: dict, overrides: list[str] | None) -> dict:
    """Применить переопределения вида 'train.max_steps=20' к словарю конфига."""
    d = copy.deepcopy(d)
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Переопределение должно быть key=value: {item!r}")
        node = d
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return d


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config_dict(path: str | os.PathLike) -> dict:
    """Читает YAML; ключ `extends: путь.yaml` (относительно файла) подмешивает базовый конфиг."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    parent = d.pop("extends", None)
    if parent:
        d = _deep_merge(load_config_dict(path.parent / parent), d)
    return d


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> Config:
    return Config.from_dict(apply_overrides(load_config_dict(path), overrides))
