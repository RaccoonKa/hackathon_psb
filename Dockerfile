FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/backend

COPY requirements.txt requirements-dev.txt /app/
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

COPY src/ /app/src/
COPY backend/ /app/backend/

COPY settings_submit.py build_index.py /app/
COPY data/ /app/data/
COPY storage/ /app/storage/

RUN mkdir -p /app/storage /app/data /app/data/raw /app/data/processed /app/data/manifests

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
