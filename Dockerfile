FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch first
RUN pip install --no-cache-dir \
    torch==2.6.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY main.py ./
COPY src/ ./src/

# HF Spaces uses port 7860
ENV PORT=7860
EXPOSE 7860

CMD python -m uvicorn main:app --host 0.0.0.0 --port $PORT
