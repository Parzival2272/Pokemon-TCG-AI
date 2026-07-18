# cabt ships libcg.so for Linux x86-64 only — build/run with:
#   docker build --platform linux/amd64 -t lucario-cabt .
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python build_deck.py

CMD ["python", "run_local.py"]
