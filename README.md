# PSB Legal Assistant — Юридический AI-ассистент

Профессиональный интеллектуальный юридический ассистент с RAG-архитектурой (Retrieval-Augmented Generation), гибридным поиском (ChromaDB Vector Store + BM25), контекстной памятью диалогов и подтверждением ответов цитатами первоисточников нормативной документации.

---

## 🚀 Быстрый старт: запуск одной командой

Проект полностью соответствует регламенту хакатона («Кейс ПСБ», слайд 11) и запускается одной стандартной командой Docker Compose с готовой предкомпилированной базой знаний:

```bash
docker compose up --build
```

- **Мгновенный старт (1–2 сек)**: Скомпилированные индексы базы знаний (`storage/chroma_db`, `storage/bm25_index`, `data/processed/chunks.jsonl`) включены в репозиторий.
- **Self-Healing (автосборка при необходимости)**: Если директория индексов очищена или монтируется пустой том, сервис автоматически выполнит первичную сборку индекса (`build_index.py`) перед стартом веб-сервера.

### Доступные интерфейсы и порты:
- **FastAPI / RAG API**: `http://localhost:8000`
- **Интерактивная документация Swagger UI**: `http://localhost:8000/docs`
- **OpenAPI спецификация**: `http://localhost:8000/openapi.json`

### Проверка работоспособности (Healthcheck):
```bash
curl -f http://localhost:8000/api/health
```
Ожидаемый ответ:
```json
{"status":"ok","rag_ready":true}
```

---

## ⚙️ Конфигурация и переменные окружения

Все переменные имеют безопасные значения по умолчанию. Для переопределения создайте файл `.env` в корне проекта (на основе `.env.example`):

| Переменная | По умолчанию | Описание |
|---|---|---|
| `OPENAI_API_KEY` / `LLM_API_KEY` | *(опционально)* | API-ключ LLM-провайдера (OpenAI / OpenRouter / GigaChat / vLLM) |
| `OPENAI_BASE_URL` / `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Базовый URL API языковой модели |
| `LLM_MODEL` | `openai/gpt-4o-mini` | Название языковой модели генерации |
| `ASSISTANT_BACKEND` | `rag` | Режим бэкенда: `rag` (полный RAG-пайплайн) или `stub` (быстрый mock для UI) |
| `DATABASE_URL` | `sqlite:///./storage/chat.db` | База данных сессий и истории диалогов пользователей |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Разрешенные источники для клиентских интерфейсов |
| `USE_LOCAL_EMBEDDING_MODEL` | `false` | `false` — удаленные API эмбеддингов; `true` — локальная `BAAI/bge-m3` |
| `ALLOW_HASH_FALLBACK` | `false` | `false` в проде (строгий контроль качества); `true` при автономной сборке без сети |
| `ENABLE_MEMORY` | `true` | Сохранение и учет контекста предыдущих реплик в диалоге |

---

## 📋 Порядок проверки основного сценария (Кейс ПСБ, слайд 11)

1. **Запуск сервиса в фоне:**
   ```bash
   docker compose up --build -d
   ```

2. **Проверка готовности RAG-системы:**
   ```bash
   curl -s http://localhost:8000/api/health
   # Возвращает {"status":"ok","rag_ready":true}
   ```

3. **Создание диалоговой сессии (чата):**
   ```bash
   CHAT_ID=$(curl -s -X POST http://localhost:8000/api/chats \
     -H "Content-Type: application/json" \
     -d '{"title": "Консультация по открытию счета"}' | grep -o '"id":"[^"]*' | cut -d'"' -f4)
   echo "ID чата: $CHAT_ID"
   ```

4. **Отправка запроса с подтверждением первоисточниками:**
   ```bash
   curl -s -X POST "http://localhost:8000/api/chats/${CHAT_ID}/messages" \
     -H "Content-Type: application/json" \
     -d '{"content": "Какие требования к согласию на обработку персональных данных?"}'
   ```
   Ответ возвращает аргументированный юридический вывод со ссылками на пункты документов и точными цитатами фрагментов текста первоисточников.

5. **Просмотр каталога нормативных документов базы знаний:**
   ```bash
   curl -s http://localhost:8000/api/documents
   ```

6. **Запуск автоматических тестов бэкенда и интеграции:**
   ```bash
   docker compose run --rm backend pytest backend/tests
   ```

7. **Остановка сервиса:**
   ```bash
   docker compose down
   ```

---

## Архитектура проекта и режимы работы

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

Скачать GGUF (выполнять в корне репозитория `hackathon-psb`):
```bash
hf download gorse-io/Qwen3-Reranker-8B-GGUF Qwen3-Reranker-8B-Q4_K_M.gguf --local-dir ./models/Qwen3-Reranker-8B
```

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


