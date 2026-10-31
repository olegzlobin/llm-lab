# llm-lab

Фреймворк на чистом PyTorch для сравнения архитектурных вариаций маленьких языковых моделей (1-3 млн параметров)
на русском корпусе историй. Обучение рассчитано на Google Colab Free (T4), локально всё проверяется на CPU.

## Данные и лицензия

Корпус: [inkoziev/ru_stories](https://huggingface.co/datasets/inkoziev/ru_stories), лицензия CC-BY-NC-4.0
(только некоммерческое использование, с указанием авторства). Данные в репозиторий не попадают:
каталоги `data/`, `checkpoints/`, `wandb/` в `.gitignore`, файлы скачиваются при запуске `prepare_data`.
Текст истории: заголовок, перевод строки, пять предложений через пробел.

## Быстрый старт (локально)

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # локально хватает CPU-сборки
uv pip install -e ".[dev]"
.venv/bin/python scripts/prepare_data.py -c configs/baseline.yaml
WANDB_MODE=disabled .venv/bin/python scripts/train.py -c configs/baseline.yaml --set train.max_steps=20
.venv/bin/python -m pytest
```

Скрипты:

| Скрипт | Назначение |
|---|---|
| `scripts/prepare_data.py` | скачивание, сплиты, дедупликация, BPE, memmap uint16 |
| `scripts/train.py` | обучение с автопродолжением (`--ckpt-dir`, `--set key=value`) |
| `scripts/measure_corpus.py` | истории и токены по сплитам для словарей 2000/4000/8000, p95 длины, размер в МБ |
| `scripts/compare_params.py` | число параметров конфигов; код возврата 1, если отличие от базы больше ±5% |
| `python -m llmlab.generate` | генерация из чекпойнта |

## Colab

Откройте `notebooks/colab_train.ipynb`, впишите `REPO_URL`, ключ W&B положите в Colab Secrets (`WANDB_API_KEY`).
Чекпойнты пишутся на Google Drive; после обрыва запустите ту же ячейку обучения: продолжение с последнего
чекпойнта и тем же W&B run происходит автоматически.

## Данные: как устроено

- Валидация (5%) выделяется из train по историям с фиксированным `split_seed`; test в обучении и отборе не используется.
- Дубли: нормализованный текст (регистр, ё/е, пробелы), совпадающий между train/val/test, удаляется из val и test
  (также дубли внутри val/test). Отчёт выводится при `prepare_data` и сохраняется в `data/splits/dup_report.json`.
- Токенизатор: byte-level BPE (`tokenizers`), обучается только на train, размер словаря `data.vocab_size`, спецтокен EOS.
- Истории конкатенируются через EOS в `data/vocab<N>/{train,val,test}.bin` (uint16). При обучении весь сплит
  загружается тензором на устройство, окна выбираются индексацией на нём (без DataLoader).

## Модель и варианты

Архитектура `gpt2` (реестр в `src/llmlab/model/registry.py`) собирается из заменяемых компонентов, выбираемых в YAML
независимо друг от друга:

| Поле | Значения |
|---|---|
| `norm` | `layernorm` (по умолчанию), `rmsnorm` |
| `ffn` | `gelu`, `swiglu` (`ffn_mult` меняет ширину) |
| `positions` | `learned`, `rope` |
| `attention` | `full`, `gqa` (+`n_kv_heads`), `sliding` (+`window_size`) |

Новая архитектура целиком: класс с `@register("имя")`, наследующий `BaseLM`, и `model.arch: имя` в YAML.
Варианты лежат в `configs/variants/` и наследуют `configs/baseline.yaml` через `extends:`.
В каждом компоненте в docstring указаны статья и отличия от оригинала.

## Правила честного сравнения

1. Один корпус, один сплит, один токенизатор (`vocab_size`) для всех вариантов; val loss сравнивается только при
   одинаковом токенизаторе.
2. Число параметров каждого варианта в пределах ±5% от базы: `python scripts/compare_params.py --base configs/baseline.yaml configs/variants/*.yaml`.
3. Одинаковые `batch_size`, `grad_accum`, `block_size`, `max_steps`, `lr`-расписание, seed и `eval_seed`
   (фиксированные окна валидации). Менять только исследуемый компонент.
4. Одинаковые ускорения (`amp_dtype`, `compile`) у всех сравниваемых запусков; фактические значения записываются
   в `metrics.json` (`runtime`) и в конфиг W&B.
5. Test используется один раз, в конце, не для выбора гиперпараметров.
6. Из-за малого числа токенов (около 4 млн в train при словаре 4000) длинные запуски означают много эпох; следите за
   разрывом train/val loss.

## Скорость

По умолчанию на CUDA: данные на GPU, autocast (bf16, если поддерживается, иначе fp16 + GradScaler),
`torch.compile` (только для шага обучения; eval и генерация идут на обычном модуле, в чекпойнте ключи без `_orig_mod.`),
fused AdamW, TF32-matmul `high`, `bias=False`, словарь дополнен до кратного 64. Лишних синхронизаций с хостом нет:
loss копится на устройстве и читается раз в `log_interval`; при NaN/inf запуск останавливается с ошибкой.
Выигрыш на T4 в этой версии не измерялся.

## Чекпойнты и W&B

`ckpt_dir/latest.pt` пишется атомарно (временный файл, fsync, `os.replace`): model, optimizer, scaler, step, все RNG
(python, numpy, torch, cuda, генератор батчей), конфиг, id запуска W&B, история. Итог: `ckpt_dir/metrics.json`.
Режим W&B: `wandb.mode` в конфиге, переменная `WANDB_MODE` приоритетнее (`disabled` для локальных запусков).
Логируются train/val loss, lr, tokens/s, пиковая память GPU, число параметров, тексты генерации (таблица).
