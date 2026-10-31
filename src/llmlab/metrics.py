"""Метрика точности для algebra_cot: жадная пакетная генерация решения и сверка ответа с answer.

Подсказка = instruction дословно ('341 - 745 = '), перед ней EOS (как в обучающем потоке, где примеры идут
через EOS; так же устроен generate_text). Генерация без KV-кэша: примеры группируются по длине подсказки,
поэтому паддинга нет. Ответ - число после последнего '= ' в последней строке сгенерированного текста.
Генерация жадная (argmax) под no_grad: ГСЧ не используется, состояние обучения не меняется.
Пример, не помещающийся в block_size (EOS + текст), считается ошибкой и не генерируется.
Если EOS не получен за лимит токенов, решение не завершено и считается ошибкой.
"""
from __future__ import annotations

import contextlib
import re
from collections import defaultdict

import torch

from .algebra import example_text, parse_instruction
from .tokenizer import eos_id

_NUM_RE = re.compile(r"-?\d+")


def parse_answer(text: str) -> int | None:
    """Число из последней строки после последнего '= '; None, если разобрать нельзя."""
    last = text.split("\n")[-1]
    if "= " not in last:
        return None
    tail = last.rsplit("= ", 1)[1].strip()
    return int(tail) if _NUM_RE.fullmatch(tail) else None


@torch.no_grad()
def greedy_batch(model, prompts: torch.Tensor, eos: int, max_new_tokens: int, check_every: int = 8,
                 amp_ctx=None) -> tuple[torch.Tensor, torch.Tensor]:
    """prompts (B,P) одинаковой длины. Возвращает (ids (B,P+n), done (B)): после EOS строка добивается EOS.
    Контекст никогда не превышает block_size (вызывающий задаёт max_new_tokens)."""
    amp_ctx = amp_ctx or contextlib.nullcontext
    V = model.cfg.vocab_size
    ids = prompts
    done = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
    for n in range(max_new_tokens):
        with amp_ctx():
            logits, _ = model(ids)
        nxt = logits[:, -1, :V].float().argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        done = done | (nxt == eos)
        if (n + 1) % check_every == 0 and bool(done.all()):
            break
    return ids, done


def _acc_table(d: dict) -> dict:
    return {k: {"n": v[0], "accuracy": v[1] / v[0]} for k, v in sorted(d.items())}


@torch.no_grad()
def evaluate_accuracy(model, tok, examples: list[dict], batch_size: int = 256, amp_ctx=None,
                      max_new_tokens: int | None = None) -> dict:
    """Точность на examples (dict: instruction, answer, reasoning). Модель - eager-модуль; режим train
    восстанавливается. Метрики: answer_accuracy, exact_solution (текст совпал с обучающим), разбивки по операции
    и по числу разрядов большего операнда."""
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        block = model.cfg.block_size
        eos = eos_id(tok)
        by_op: dict = defaultdict(lambda: [0, 0])
        by_dig: dict = defaultdict(lambda: [0, 0])
        tot = {"n": 0, "answer": 0, "exact": 0, "too_long": 0, "no_eos": 0}

        def record(ex, ok_answer, ok_exact):
            a, op, b = parse_instruction(ex["instruction"])
            dig = max(len(str(abs(a))), len(str(abs(b))))
            for tab, key in ((by_op, op), (by_dig, dig)):
                tab[key][0] += 1
                tab[key][1] += int(ok_answer)
            tot["n"] += 1
            tot["answer"] += int(ok_answer)
            tot["exact"] += int(ok_exact)

        groups: dict[int, list] = defaultdict(list)  # длина подсказки -> [(ex, prompt_ids)]
        for ex in examples:
            ref_len = len(tok.encode(example_text(ex)).ids)
            prompt = tok.encode(ex["instruction"]).ids
            if 1 + ref_len > block:  # EOS + текст не помещаются в контекст: ошибка, без генерации
                tot["too_long"] += 1
                record(ex, False, False)
            elif 1 + len(prompt) > block:
                tot["too_long"] += 1
                record(ex, False, False)
            else:
                groups[len(prompt)].append((ex, prompt))

        for plen, items in sorted(groups.items()):
            limit = block - (plen + 1) + 1  # последний токен (EOS) предсказывается по контексту <= block_size
            if max_new_tokens is not None:
                limit = min(limit, max_new_tokens)
            for s in range(0, len(items), batch_size):
                chunk = items[s:s + batch_size]
                pr = torch.tensor([[eos] + p for _, p in chunk], dtype=torch.long, device=device)
                ids, _ = greedy_batch(model, pr, eos, limit, amp_ctx=amp_ctx)
                rows = ids[:, 1:].tolist()  # без ведущего EOS
                for (ex, _), row in zip(chunk, rows):
                    if eos in row:
                        row = row[:row.index(eos)]
                        finished = True
                    else:
                        finished = False
                    text = tok.decode(row)
                    if not finished:
                        tot["no_eos"] += 1
                        record(ex, False, False)
                        continue
                    ans = parse_answer(text)
                    record(ex, ans is not None and ans == int(ex["answer"]), text == example_text(ex))
    finally:
        model.train(was_training)
    n = max(tot["n"], 1)
    return {
        "n": tot["n"], "answer_accuracy": tot["answer"] / n, "exact_solution": tot["exact"] / n,
        "too_long": tot["too_long"], "no_eos": tot["no_eos"],
        "by_op": _acc_table(by_op), "by_digits": _acc_table(by_dig),
    }


def flat_for_log(res: dict, prefix: str) -> dict:
    """Плоский словарь для W&B."""
    out = {f"{prefix}/answer_accuracy": res["answer_accuracy"], f"{prefix}/exact_solution": res["exact_solution"]}
    for k, v in res["by_op"].items():
        out[f"{prefix}/op_{'plus' if k == '+' else 'minus'}"] = v["accuracy"]
    for k, v in res["by_digits"].items():
        out[f"{prefix}/digits_{k}"] = v["accuracy"]
    return out


def format_table(res: dict) -> str:
    L = [f"примеров: {res['n']}  answer_accuracy: {res['answer_accuracy']:.4f}  "
         f"exact_solution: {res['exact_solution']:.4f}  не вошли в block_size (ошибка): {res['too_long']}  "
         f"без EOS (ошибка): {res['no_eos']}", "",
         "| группа | значение | примеров | answer_accuracy |", "|---|---|---|---|"]
    for k, v in res["by_op"].items():
        L.append(f"| операция | {k} | {v['n']} | {v['accuracy']:.4f} |")
    for k, v in res["by_digits"].items():
        L.append(f"| разрядов (больший операнд) | {k} | {v['n']} | {v['accuracy']:.4f} |")
    return "\n".join(L)
