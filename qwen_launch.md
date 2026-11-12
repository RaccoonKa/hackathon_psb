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
