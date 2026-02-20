FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY configs.py engines.py audio.py llm.py main.py ./

# HF Spaces uses port 7860, override with PORT env var
ENV PORT=7860
EXPOSE 7860

# Model is downloaded on first run via configs.download_model()
CMD python -m uvicorn main:app --host 0.0.0.0 --port $PORT
