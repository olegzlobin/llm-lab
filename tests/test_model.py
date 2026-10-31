import math

import pytest
import torch

from conftest import tiny_config
from llmlab.model import available, build_model, register
from llmlab.model.base import BaseLM
from llmlab.model.components import Attention, RMSNorm, apply_rope, rope_cache

VARIANTS = {
    "base": {},
    "rmsnorm": {"norm": "rmsnorm"},
    "swiglu": {"ffn": "swiglu"},
    "rope": {"positions": "rope"},
    "gqa": {"attention": "gqa", "n_kv_heads": 1},
    "sliding": {"attention": "sliding", "window_size": 8},
    "all": {"norm": "rmsnorm", "ffn": "swiglu", "positions": "rope", "attention": "gqa", "n_kv_heads": 1},
}


def build(**kw):
    torch.manual_seed(0)
    cfg = tiny_config(dropout=0.0, **kw).model
    return build_model(cfg).eval(), cfg


@pytest.mark.parametrize("name", VARIANTS)
def test_shapes_and_causality(name):
    m, cfg = build(**VARIANTS[name])
    x = torch.randint(0, 300, (3, 20))
    logits, loss = m(x, x)
    assert logits.shape == (3, 20, m.padded_vocab) and m.padded_vocab == 320
    assert loss.ndim == 0
    x2 = x.clone()
    x2[:, 12:] = torch.randint(0, 300, (3, 8))  # меняем будущее
    l2, _ = m(x2)
    assert torch.allclose(logits[:, :12], l2[:, :12], atol=1e-5)
    assert not torch.allclose(logits[:, 12:], l2[:, 12:], atol=1e-5)


def test_weight_tying():
    m, _ = build()
    assert m.lm_head.weight is m.wte.weight
    names = [n for n, _ in m.named_parameters()]
    assert not any("lm_head" in n for n in names)


@pytest.mark.parametrize("name", VARIANTS)
def test_initial_loss_is_ln_vocab(name):
    m, _ = build(**VARIANTS[name])
    x = torch.randint(0, 300, (16, 32))
    y = torch.randint(0, 300, (16, 32))  # независимые цели (при tying цель=вход даёт loss < ln V)
    _, loss = m(x, y)
    assert abs(loss.item() - math.log(m.padded_vocab)) < 0.1


def test_residual_init_scaling():
    m, cfg = build()
    std = m.blocks[0].attn.proj.weight.std().item()
    assert abs(std - 0.02 / math.sqrt(2 * cfg.n_layer)) < 0.004
    assert abs(m.blocks[0].attn.qkv.weight.std().item() - 0.02) < 0.004


@pytest.mark.parametrize("name", VARIANTS)
def test_overfit_tiny_batch(name):
    torch.manual_seed(0)
    cfg = tiny_config(dropout=0.0, **VARIANTS[name]).model
    m = build_model(cfg).train()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, 300, (4, 16))
    y = torch.randint(0, 300, (4, 16))
    for _ in range(150):
        _, loss = m(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.15


def test_rope_preserves_norm_and_relative():
    cos, sin = rope_cache(16, 32)
    x = torch.randn(2, 3, 32, 16)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)
    # скалярное произведение зависит только от разности позиций
    q, k = torch.randn(16), torch.randn(16)
    def dot(i, j):
        qi = apply_rope(q.expand(1, 1, 32, 16).contiguous(), cos, sin)[0, 0, i]
        kj = apply_rope(k.expand(1, 1, 32, 16).contiguous(), cos, sin)[0, 0, j]
        return (qi * kj).sum()
    assert torch.allclose(dot(5, 2), dot(15, 12), atol=1e-4)


def test_rmsnorm_values():
    n = RMSNorm(8)
    x = torch.randn(4, 8)
    y = n(x)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4), atol=1e-3)


def test_gqa_with_full_kv_heads_equals_full():
    torch.manual_seed(1)
    full = build_model(tiny_config(dropout=0.0).model).eval()
    cfg = tiny_config(dropout=0.0, attention="gqa", n_kv_heads=2).model  # n_head=2
    gqa = build_model(cfg).eval()
    assert full.num_params() == gqa.num_params()
    gqa.load_state_dict(full.state_dict())  # одинаковая раскладка весов
    x = torch.randint(0, 300, (2, 24))
    assert torch.allclose(full(x)[0], gqa(x)[0], atol=1e-6)


def test_gqa_reduces_params_and_validates():
    a, _ = build()
    b, _ = build(attention="gqa", n_kv_heads=1)
    assert b.num_params() < a.num_params()
    with pytest.raises(ValueError):
        build(attention="gqa")
    with pytest.raises(ValueError):
        build(attention="gqa", n_kv_heads=3)


def test_sliding_window_blindness():
    cfg = tiny_config(dropout=0.0, attention="sliding", window_size=4, n_layer=1).model
    torch.manual_seed(0)
    m = build_model(cfg).eval()
    x = torch.randint(0, 300, (1, 20))
    base = m(x)[0]
    x2 = x.clone()
    x2[0, 5] = (x2[0, 5] + 1) % 300  # позиция 5 вне окна для запросов >= 9
    out = m(x2)[0]
    assert torch.allclose(base[0, :5], out[0, :5], atol=1e-6)
    assert torch.allclose(base[0, 9:], out[0, 9:], atol=1e-5)  # окно 4: 9 видит 6..9
    assert not torch.allclose(base[0, 5:9], out[0, 5:9], atol=1e-5)  # 5..8 видят позицию 5


def test_registry():
    assert "gpt2" in available()
    with pytest.raises(KeyError):
        build_model(tiny_config(arch="nope").model)

    @register("dummy_test_arch")
    class Dummy(BaseLM):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.p = torch.nn.Parameter(torch.zeros(3))

    try:
        assert build_model(tiny_config(arch="dummy_test_arch").model).num_params() == 3
    finally:
        from llmlab.model import registry
        registry._REGISTRY.pop("dummy_test_arch")
