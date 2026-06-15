# Match Ray head image exactly: same Ray version AND Python 3.11.8
FROM rayproject/ray:2.11.0-py311

RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    build-essential git curl \
    && sudo rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# PyTorch (CUDA 12.1) + remaining training deps (ray already in base image)
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt

COPY train.py .

ENV PYTHONUNBUFFERED=1
EXPOSE 8003

CMD ["python", "train.py"]
