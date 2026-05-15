FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY frontend/ ./frontend/

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["doc2graph", "serve", "--host", "0.0.0.0", "--port", "8000"]
