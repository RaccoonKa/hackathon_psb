hack_psb/
│
├── data/
│   ├── raw/                  # Сюда копируем исходную базу знаний (knowledge-base)
│   └── processed/            # Промежуточный JSON со структурированными чанками и метаданными
│
├── storage/
│   ├── chroma_db/            # Готовый персистентный векторный индекс ChromaDB (сдается в ZIP)
│   └── bm25_index/           # Сохраненный индекс BM25 (bm25s) для гибридного поиска
│
├── src/
│   ├── __init__.py
│   ├── config.py             # Пути, имена моделей (bge-m3, DeepSeek V4.1), гиперпараметры
│   │
│   ├── indexing/             # Модули подготовки базы (запуск один раз офлайн)
│   │   ├── __init__.py
│   │   ├── parser.py         # Извлечение структуры: названий, глав, статей, дат и редакций
│   │   └── chunker.py        # Нарезка по статьям/пунктам без разрыва юридического контекста
│   │
│   ├── retrieval/            # Модули поиска
│   │   ├── __init__.py
│   │   ├── vector_store.py   # Работа с ChromaDB (BAAI/bge-m3)
│   │   ├── keyword_store.py  # BM25-поиск по точным юридическим формулировкам и номерам статей
│   │   └── hybrid.py         # Слияние результатов (Ensemble / Reranker Qwen3)
│   │
│   ├── generation/           # Модули ответа и контроля галлюцинаций
│   │   ├── __init__.py
│   │   ├── prompts.py        # Системные промпты (отказы, даты, защита от инъекций)
│   │   └── rag_chain.py      # Вызов DeepSeek V4.1 Flash, самопроверка (Self-Check)
│   │
│   └── web/                  # Веб-интерфейс чата под требования сдачи
│       ├── __init__.py
│       ├── api.py            # Эндпоинты FastAPI (/ask, /documents/{id})
│       └── static/           # Статический HTML/JS фронтенд чата с боковой панелью цитат
│
├── build_index.py            # Офлайн-скрипт: парсит data/raw -> создает индексы в storage/
├── solution.py               # Точка входа для автопроверки LLM-судьей (чтение CSV -> прогон -> запись)
├── main.py                   # Запуск веб-сервера FastAPI (uvicorn main:app)
│
├── Dockerfile                # Развертывание одной командой
├── compose.yaml              # Docker Compose для запуска
├── .dockerignore
├── .env.example              # Шаблон конфигурации API
└── requirements.txt          # Зафиксированные версии библиотек