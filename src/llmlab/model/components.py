"""Заменяемые компоненты трансформера: нормализация, FFN, RoPE, внимание (full/GQA/sliding)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


def round8(x: float) -> int:
    return max(8, int(round(x / 8)) * 8)


# --- нормализация ---------------------------------------------------------------
class RMSNorm(nn.Module):
    """RMSNorm: y = x / sqrt(mean(x^2) + eps) * g.

    Статья: Zhang & Sennrich, "Root Mean Square Layer Normalization", 2019 (arXiv:1910.07467).
    Отличия от оригинала: нет смещения; в оригинале есть необязательный pRMSNorm (часть признаков),
    здесь только полный RMS; вычисление в float32 для устойчивости под AMP.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def make_norm(kind: str, dim: int, bias: bool) -> nn.Module:
    if kind == "layernorm":
        return nn.LayerNorm(dim, bias=bias)
    if kind == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Неизвестная norm {kind!r} (layernorm | rmsnorm)")


# --- FFN ------------------------------------------------------------------------
class GeluFFN(nn.Module):
    """Классический FFN GPT-2: Linear -> GELU -> Linear (Radford et al., 2019).

    Отличие: GELU в tanh-приближении как в GPT-2; bias управляется конфигом (по умолчанию выключен).
    """

    def __init__(self, cfg: ModelConfig, hidden: int):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class SwiGLU(nn.Module):
    """SwiGLU FFN: proj(silu(x W_gate) * (x W_up)).

    Статья: Shazeer, "GLU Variants Improve Transformer", 2020 (arXiv:2002.05202).
    Отличия: bias по конфигу (в статье нет); gate и up объединены в один Linear; скрытая ширина
    по умолчанию 8/3*n_embd (кратно 8), чтобы число параметров было близко к GELU-FFN с 4*n_embd.
    """

    def __init__(self, cfg: ModelConfig, hidden: int):
        super().__init__()
        self.gate_up = nn.Linear(cfg.n_embd, 2 * hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.drop(self.proj(F.silu(gate) * up))


def make_ffn(cfg: ModelConfig) -> nn.Module:
    if cfg.ffn == "gelu":
        return GeluFFN(cfg, round8((cfg.ffn_mult or 4.0) * cfg.n_embd))
    if cfg.ffn == "swiglu":
        return SwiGLU(cfg, round8((cfg.ffn_mult or 8.0 / 3.0) * cfg.n_embd))
    raise ValueError(f"Неизвестный ffn {cfg.ffn!r} (gelu | swiglu)")


# --- RoPE -----------------------------------------------------------------------
def rope_cache(head_dim: int, max_len: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(max_len).float(), inv)  # (max_len, hd/2)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary Position Embedding: вращает пары каналов (i, i+hd/2) на угол pos*theta_i.

    Статья: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
    (arXiv:2104.09864). Отличия: раскладка пар "половинами" (как в GPT-NeoX/LLaMA), а не соседними
    каналами; база 10000, без масштабирования длины; вращаются только q и k (не v). x: (B,H,T,hd).
    """
    T, half = x.shape[-2], x.shape[-1] // 2
    cos, sin = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


# --- внимание -------------------------------------------------------------------
class Attention(nn.Module):
    """Причинное многоголовое внимание на F.scaled_dot_product_attention.

    Режимы (cfg.attention):
      full    - стандартное внимание GPT-2 (Vaswani et al., 2017; Radford et al., 2019), is_causal=True.
      gqa     - Grouped-Query Attention: n_kv_heads < n_head, группа q-голов делит одну K/V-голову.
                Статья: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from
                Multi-Head Checkpoints", 2023 (arXiv:2305.13245). Отличие: обучаем с нуля, без uptraining;
                K/V расширяются repeat_interleave (без KV-кэша, т.к. генерация без кэша).
      sliding - локальное причинное окно window_size (токен i видит i-w+1..i), как в Longformer
                (Beltagy et al., 2020) / Mistral (Jiang et al., 2023), но без глобальных токенов и без
                rolling-буфера. Маска предвычислена один раз (буфер), на шаге только срез.
    Один fused Linear на q,k,v: при n_kv_heads=n_head раскладка весов совпадает с full.
    Позиции RoPE применяются здесь, если cfg.positions == "rope".
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        H = cfg.n_head
        if cfg.n_embd % H:
            raise ValueError("n_embd должно делиться на n_head")
        self.n_head, self.hd, self.dropout = H, cfg.n_embd // H, cfg.dropout
        kv_heads = cfg.n_kv_heads or H
        if cfg.attention == "full":
            if kv_heads != H:
                raise ValueError("attention=full не совместим с n_kv_heads != n_head; используйте gqa")
        elif cfg.attention == "gqa":
            if cfg.n_kv_heads is None:
                raise ValueError("attention=gqa требует n_kv_heads")
        elif cfg.attention == "sliding":
            if not cfg.window_size or cfg.window_size < 1:
                raise ValueError("attention=sliding требует window_size >= 1")
        else:
            raise ValueError(f"Неизвестное attention {cfg.attention!r} (full | gqa | sliding)")
        if H % kv_heads:
            raise ValueError("n_head должно делиться на n_kv_heads")
        self.n_kv = kv_heads
        kv_dim = kv_heads * self.hd
        self.qkv = nn.Linear(cfg.n_embd, cfg.n_embd + 2 * kv_dim, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.window = cfg.window_size if cfg.attention == "sliding" else None
        self.rope = cfg.positions == "rope"
        if self.rope:
            if self.hd % 2:
                raise ValueError("RoPE требует чётную размерность головы")
            cos, sin = rope_cache(self.hd, cfg.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        if self.window:
            i = torch.arange(cfg.block_size)
            mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < self.window)  # True = можно смотреть
            self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        kv_dim = self.n_kv * self.hd
        q, k, v = self.qkv(x).split([C, kv_dim, kv_dim], dim=-1)
        q = q.view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_kv, self.hd).transpose(1, 2)
        if self.rope:
            q, k = apply_rope(q, self.rope_cos, self.rope_sin), apply_rope(k, self.rope_cos, self.rope_sin)
        if self.n_kv != self.n_head:
            rep = self.n_head // self.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
        dp = self.dropout if self.training else 0.0
        if self.window:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask[:T, :T], dropout_p=dp)
        else:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=dp, is_causal=True)
        return self.resid_drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class Block(nn.Module):
    """Pre-norm блок: x + attn(norm(x)); x + ffn(norm(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = make_norm(cfg.norm, cfg.n_embd, cfg.bias)
        self.attn = Attention(cfg)
        self.norm2 = make_norm(cfg.norm, cfg.n_embd, cfg.bias)
        self.ffn = make_ffn(cfg)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


__all__ = ["RMSNorm", "GeluFFN", "SwiGLU", "Attention", "Block", "make_norm", "make_ffn", "rope_cache",
           "apply_rope", "round8", "math"]
