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

--- 

# Qwen3-Reranker-8B GGUF — запуск на macOS, Linux и Windows

Инструкция по запуску локального reranker-сервера на базе `llama.cpp` и подключению его к RAG-проекту.

## 1. Что понадобится

- `llama.cpp` с поддержкой reranking.
- GGUF-файл модели:
  `Qwen3-Reranker-8B-Q4_K_M.gguf`
- Python-окружение проекта.
- Свободный локальный порт `8081`.

Дополнительные файлы `conversion.json` и `conversion_adapter.py` для запуска готового GGUF не требуются.

Рекомендуемая структура проекта:

```text
project/
├── models/
│   └── Qwen3-Reranker-8B/
│       └── Qwen3-Reranker-8B-Q4_K_M.gguf
├── src/
├── .env
└── run_live.py
```

Если GGUF уже скачан, повторно загружать модель не нужно.

---

## 2. Установка llama.cpp

### macOS

Установка через Homebrew:

```bash
brew install llama.cpp
```

Проверка:

```bash
llama-server --version
```

На Mac с Apple Silicon llama.cpp может использовать Metal для ускорения вычислений.

### Linux

Рекомендуемый вариант — официальный установщик:

```bash
curl -LsSf https://llama.app/install.sh | sh
```

Альтернативный вариант через Homebrew:

```bash
brew install llama.cpp
```

Проверка:

```bash
llama-server --version
```

Если команда не найдена, проверь PATH и расположение установленного исполняемого файла.

### Windows

Открой PowerShell и выполни:

```powershell
winget install llama.cpp
```

После установки открой новое окно PowerShell.

Проверка:

```powershell
llama-server --version
```

Если команда не найдена, проверь, что каталог с `llama-server.exe` добавлен в PATH.

---

## 3. Запуск модели

Перед запуском перейди в корневую папку проекта.

### macOS и Linux

```bash
llama-server \
  -m models/Qwen3-Reranker-8B/Qwen3-Reranker-8B-Q4_K_M.gguf \
  --embedding \
  --reranking \
  --pooling rank \
  --host 127.0.0.1 \
  --port 8081 \
  --alias qwen3-reranker-8b
```

### Windows — PowerShell

```powershell
llama-server.exe `
  -m "models\Qwen3-Reranker-8B\Qwen3-Reranker-8B-Q4_K_M.gguf" `
  --embedding `
  --reranking `
  --pooling rank `
  --host 127.0.0.1 `
  --port 8081 `
  --alias qwen3-reranker-8b
```

Команды запускают локальный HTTP-сервер:

```text
http://127.0.0.1:8081
```

**Важно:** оставь это окно терминала открытым. Пока сервер работает, он принимает запросы от RAG-проекта.

Если модель не помещается в доступную память, попробуй уменьшить нагрузку или использовать другую квантованную версию модели.


