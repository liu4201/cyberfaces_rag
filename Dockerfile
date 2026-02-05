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

COPY requirements.txt .
COPY data.jsonl .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Expose FastAPI default port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
