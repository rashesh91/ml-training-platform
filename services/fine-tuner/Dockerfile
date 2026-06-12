FROM python:3.11-slim

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Install PyTorch with CUDA 12.1 support first, then remaining deps
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

COPY train.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 8003

CMD ["python", "train.py"]
