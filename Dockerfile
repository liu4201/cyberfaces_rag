FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
COPY data.jsonl .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Expose FastAPI default port
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
