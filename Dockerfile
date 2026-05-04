# syntax=docker/dockerfile:1
FROM python:3.12-slim

# 1. Install build tools and C++ compilers
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    build-essential \
#    gcc \
#    g++ \
#    python3-dev \
#    && rm -rf /var/lib/apt/lists/*

# 2. Crucial: Ensure pip is updated (helps with ARM64 wheel discovery)
#RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# Copy only requirements first for better layer caching
COPY requirements.txt .
COPY data.jsonl .
COPY gemini_generate_dataset_updateByHuman.jsonl .

# Install dependencies (this layer will be cached unless requirements.txt changes)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy application code after dependencies are installed
COPY app ./app
RUN mkdir -p ./chromaDB

# Expose FastAPI default port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
