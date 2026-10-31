"""Реестр архитектур: новый вариант = один класс с @register("имя"), выбор через model.arch в YAML."""
from __future__ import annotations

from ..config import ModelConfig
from .base import BaseLM

_REGISTRY: dict[str, type[BaseLM]] = {}


def register(name: str):
    def deco(cls):
        if name in _REGISTRY:
            raise ValueError(f"Архитектура {name!r} уже зарегистрирована")
        _REGISTRY[name] = cls
        cls.arch_name = name
        return cls

    return deco


def available() -> list[str]:
    return sorted(_REGISTRY)


def build_model(cfg: ModelConfig) -> BaseLM:
    if cfg.arch not in _REGISTRY:
        raise KeyError(f"Неизвестная архитектура {cfg.arch!r}; доступны: {available()}")
    return _REGISTRY[cfg.arch](cfg)
