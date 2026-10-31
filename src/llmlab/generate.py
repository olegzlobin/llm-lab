"""Генерация текста. CLI: python -m llmlab.generate --ckpt checkpoints/baseline/latest.pt --prompt "Новый год\n" """
from __future__ import annotations

import argparse

import torch

from .tokenizer import eos_id


@torch.no_grad()
def generate_ids(model, ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, generator: torch.Generator | None = None, eos: int | None = None):
    """ids: (1,T). Без KV-кэша; контекст обрезается до block_size. Останов по EOS."""
    was_training = model.training
    model.eval()
    block = model.cfg.block_size
    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -block:])
        logits = logits[:, -1, : model.cfg.vocab_size].float()  # без строк паддинга словаря
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
        ids = torch.cat([ids, nxt], dim=1)
        if eos is not None and nxt.item() == eos:
            break
    model.train(was_training)
    return ids


def generate_text(model, tok, prompt: str, max_new_tokens=100, temperature=0.8, top_k=40, seed=0, device="cpu") -> str:
    eos = eos_id(tok)
    gen = torch.Generator(device=device).manual_seed(seed)
    ids = torch.tensor([[eos] + tok.encode(prompt).ids], device=device)
    out = generate_ids(model, ids, max_new_tokens, temperature, top_k, gen, eos)
    return tok.decode([i for i in out[0].tolist() if i != eos])


def main() -> None:
    from .checkpoint import load_checkpoint
    from .config import Config
    from .data import corpus_dir, load_corpus_tokenizer
    from .model import build_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Новый год\n")
    ap.add_argument("--data-dir", default=None, help="корень данных (по умолчанию из конфига чекпойнта)")
    ap.add_argument("--max-new-tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ck = load_checkpoint(a.ckpt)
    cfg = Config.from_dict(ck["config"])
    model = build_model(cfg.model)
    model.load_state_dict(ck["model"])
    tok = load_corpus_tokenizer(corpus_dir(cfg, a.data_dir))
    prompt = a.prompt.replace("\\n", "\n")
    print(generate_text(model, tok, prompt, a.max_new_tokens, a.temperature, a.top_k, a.seed))


if __name__ == "__main__":
    main()
