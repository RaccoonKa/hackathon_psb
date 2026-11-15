FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/backend

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY requirements.txt requirements-dev.txt /app/

RUN uv pip install --system --no-cache -r requirements.txt -r requirements-dev.txt

COPY rag_core/ /app/rag_core/
COPY backend/ /app/backend/

COPY settings_submit.py build_index.py /app/
COPY data/ /app/data/
COPY storage/ /app/storage/

RUN mkdir -p /app/storage /app/data /app/data/raw /app/data/processed /app/data/manifests

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
