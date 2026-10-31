"""BPE-токенизатор (библиотека tokenizers), byte-level, обучается только на train."""
from __future__ import annotations

from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS_TOKEN = "<|endoftext|>"


def train_tokenizer(texts: Iterable[str], vocab_size: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOS_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def load_tokenizer(path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def eos_id(tok: Tokenizer) -> int:
    i = tok.token_to_id(EOS_TOKEN)
    assert i is not None, "в токенизаторе нет EOS"
    return i


def encode_texts(tok: Tokenizer, texts: list[str]) -> list[list[int]]:
    return [e.ids for e in tok.encode_batch(texts)]
