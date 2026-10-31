"""Архитектура "gpt2": GPT-2-подобный decoder-only трансформер из заменяемых компонентов."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .base import BaseLM
from .components import Block, make_norm
from .registry import register

PAD_VOCAB_TO = 64


@register("gpt2")
class Transformer(BaseLM):
    """GPT-2: pre-LN, learned positional embeddings, LayerNorm, GELU, причинное внимание, weight tying.

    Статья: Radford et al., "Language Models are Unsupervised Multitask Learners", 2019 (GPT-2);
    основа - Vaswani et al., "Attention Is All You Need", 2017.
    Отличия от оригинала: bias по умолчанию выключен; словарь дополняется до кратного 64 (лишние
    строки эмбеддинга никогда не являются целями); размеры n_embd/block_size кратны 8 (Tensor Cores);
    инициализация N(0, 0.02), residual-проекции масштабируются std = 0.02/sqrt(2*n_layer).
    При конфиге по умолчанию (layernorm/gelu/learned/full) это классический GPT-2; компоненты norm,
    ffn, positions, attention независимо заменяются из YAML (см. model/components.py).
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        if cfg.n_embd % 8 or cfg.block_size % 8:
            raise ValueError("n_embd и block_size должны быть кратны 8")
        self.padded_vocab = math.ceil(cfg.vocab_size / PAD_VOCAB_TO) * PAD_VOCAB_TO
        self.wte = nn.Embedding(self.padded_vocab, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.positions == "learned" else None
        if cfg.positions not in ("learned", "rope"):
            raise ValueError(f"Неизвестные positions {cfg.positions!r} (learned | rope)")
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = make_norm(cfg.norm, cfg.n_embd, cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, self.padded_vocab, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying
        self.register_buffer("pos_ids", torch.arange(cfg.block_size), persistent=False)

        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):  # выходные проекции attention и FFN
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"Длина {T} больше block_size={self.cfg.block_size}")
        x = self.wte(idx)
        if self.wpe is not None:
            x = x + self.wpe(self.pos_ids[:T])
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
