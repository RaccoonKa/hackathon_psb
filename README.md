# hackathon-psb

Юридический RAG-ассистент с единой бизнес-логикой и двумя режимами:
локальной разработкой и платформенным сабмитом.

## Переключатель режимов

Единственный ручной переключатель находится в [`settings_submit.py`](settings_submit.py:1):

```python
USE_SUBMIT = False
```

- `False` — LOCAL: локальные `data/`, `storage/`, `.env`, отдельные URL/ключи и
  локальная модель `models/bge-m3` (если настроена).
- `True` — SUBMIT: платформенные `DATA_DIR`, единый прокси `LLM_BASE_URL`/
  `LLM_API_KEY`, записываемый временный рабочий каталог, без локальных весов и
  без hash-fallback эмбеддингов.

Все модули читают настройки только через [`src/config.py`](src/config.py:1).

## Локальная разработка

```bash
python run_live.py
python build_index.py
```

## Сборка сабмита

```bash
python build_submit.py
```

Скрипт полностью пересоздаёт [`submit/`](submit:1), переписывает импорты
`src.*` в плоские имена через AST, формирует `submit/settings_submit.py` со
значением `USE_SUBMIT = True` (исходный файл не меняется), создаёт
`requirements.txt` и `README.md`, затем запускает проверки.

Запуск собранного артефакта:

```bash
cd submit
python solution.py
```

В текущем рабочем каталоге создаётся `submission.csv` с колонками `id,answer`.

ZIP с файлами в корне архива:

```bash
python build_submit.py --zip
```

## Проверки

```bash
python verify_submit.py --submit-dir submit
```

Проверяются плоская структура, отсутствие `src.*`, изолированный импорт,
конфигурация LOCAL/SUBMIT, безопасная распаковка ZIP, чтение TXT/MD/DOCX,
контракт CSV, однократная индексация и отсутствие неполного сабмита при ошибке.

## Ограничения

`submit/requirements.txt` содержит зафиксированные runtime-зависимости проекта
(`chromadb`, `langchain-openai`, `langchain-text-splitters`, `openai`,
`rank-bm25`, `requests`, `tiktoken`). Они присутствуют в базовом
`requirements.txt` проекта; если allowlist платформы окажется уже, набор
зависимостей нужно согласовать до отправки.
