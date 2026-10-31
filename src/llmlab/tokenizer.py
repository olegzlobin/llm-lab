"""Токенизаторы: BPE (библиотека tokenizers, byte-level) и посимвольный; обучаются только на train."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOS_TOKEN = "<|endoftext|>"
EOS_SENTINEL = "\x00"  # служебный разделитель текстов при быстром кодировании; в данных запрещён


@dataclass
class _Encoding:
    ids: list[int]


class CharTokenizer:
    """Посимвольный токенизатор с тем же интерфейсом, что нужен проекту от tokenizers.Tokenizer:
    save, from_file (через load_tokenizer), encode(...).ids, encode_batch, decode, token_to_id, get_vocab_size.

    id 0 = EOS, символы словаря получают id 1..N по порядку кодовых точек. Словарь строится только из train.
    Символ вне словаря даёт UnknownCharError с перечнем символов (молча заменять нельзя: это скрыло бы
    утечку/несовпадение данных между сплитами)."""

    def __init__(self, chars: list[str]):
        if len(set(chars)) != len(chars) or any(len(c) != 1 for c in chars):
            raise ValueError("chars: ожидаются уникальные односимвольные строки")
        if chars != sorted(chars):
            raise ValueError("chars должны быть отсортированы по кодовой точке")
        if EOS_SENTINEL in chars:
            raise ValueError("символ \\x00 зарезервирован под служебные нужды")
        self.chars = list(chars)
        self._stoi = {c: i + 1 for i, c in enumerate(self.chars)}
        cps = np.array([ord(c) for c in self.chars], dtype=np.int64)  
        self._cps = cps

    @classmethod
    def train(cls, texts: Iterable[str]) -> "CharTokenizer":
        chars: set[str] = set()
        for t in texts:
            chars.update(t)
        return cls(sorted(chars))

    def get_vocab_size(self) -> int:
        return len(self.chars) + 1

    def token_to_id(self, token: str) -> int | None:
        if token == EOS_TOKEN:
            return 0
        return self._stoi.get(token)

    def encode(self, text: str) -> _Encoding:
        try:
            return _Encoding([self._stoi[c] for c in text])
        except KeyError:
            raise UnknownCharError(sorted({c for c in text if c not in self._stoi})) from None

    def encode_batch(self, texts: list[str]) -> list[_Encoding]:
        return [self.encode(t) for t in texts]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == 0:
                if not skip_special_tokens:
                    out.append(EOS_TOKEN)
            elif 1 <= i <= len(self.chars):
                out.append(self.chars[i - 1])
            else:
                raise ValueError(f"id {i} вне словаря (размер {self.get_vocab_size()})")
        return "".join(out)

    def encode_stream(self, texts: list[str], chunk: int = 20000) -> np.ndarray:
        """Тексты подряд, после каждого EOS (id 0): int64-вектор для сплита. Быстро: через кодовые точки."""
        parts = []
        for s in range(0, len(texts), chunk):
            part = texts[s:s + chunk]
            joined = EOS_SENTINEL.join(part) + EOS_SENTINEL
            if any(EOS_SENTINEL in t for t in part):
                raise ValueError("в тексте есть служебный символ \\x00")
            cp = np.frombuffer(joined.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
            pos = np.searchsorted(self._cps, cp)
            pos_c = np.minimum(pos, len(self._cps) - 1)
            known = (self._cps[pos_c] == cp) if len(self._cps) else np.zeros_like(cp, dtype=bool)
            is_eos = cp == ord(EOS_SENTINEL)
            bad = ~(known | is_eos)
            if bad.any():
                raise UnknownCharError(sorted({chr(int(c)) for c in cp[bad]}))
            ids = np.where(is_eos, 0, pos_c + 1)
            parts.append(ids)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)

    def save(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "eos_token": EOS_TOKEN, "chars": self.chars}, f, ensure_ascii=False)

    @classmethod
    def from_file(cls, path) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["chars"])


class UnknownCharError(ValueError):
    def __init__(self, chars: list[str]):
        self.chars = chars
        super().__init__(f"символы вне словаря посимвольного токенизатора (словарь построен по train): {chars[:20]!r}"
                         f"{' ...' if len(chars) > 20 else ''}")


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


def train_char_tokenizer(texts: Iterable[str]) -> CharTokenizer:
    return CharTokenizer.train(texts)


def load_tokenizer(path):
    """BPE (tokenizer.json библиотеки tokenizers) или посимвольный ({"type": "char", ...})."""
    with open(path, encoding="utf-8") as f:
        if json.load(f).get("type") == "char":
            return CharTokenizer.from_file(path)
    return Tokenizer.from_file(str(path))


def eos_id(tok) -> int:
    i = tok.token_to_id(EOS_TOKEN)
    assert i is not None, "в токенизаторе нет EOS"
    return i


def encode_texts(tok, texts: list[str]) -> list[list[int]]:
    return [e.ids for e in tok.encode_batch(texts)]
