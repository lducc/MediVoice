---
title: MediVoice
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# MediVoice API

MediVoice is a medical voice transcription system with speaker diarization and LLM-assisted Electronic Medical Record (EMR) extraction. It converts doctor-patient conversations into structured medical data using state-of-the-art AI models.

**[Try it live on HuggingFace Spaces →](https://huggingface.co/spaces/lducc/MediVoice)**

## Features

- **Audio Transcription**: Converts audio to text using faster-whisper (Vietnamese language)
- **Speaker Diarization**: Identifies and separates different speakers (doctor vs patient)
- **EMR Extraction**: Automatically extracts structured medical data from conversations using LLM
- **Fast Processing**: GPU-accelerated with batched inference (CTranslate2)
- **Interactive UI**: Built-in Gradio interface for easy interaction
- **REST API**: FastAPI endpoints for programmatic access

## Demo

The app provides a Gradio web interface with two tabs:

| Tab | Description |
|---|---|
| **Speech to Text** | Upload audio → get transcript with optional speaker labels |
| **EMR Extraction** | Paste transcript → get structured medical JSON |

## Models

| Component | Model | Link |
|---|---|---|
| ASR | Whisper (Vietnamese fine-tuned, CTranslate2) | [lducc/MediVoice-ct2](https://huggingface.co/lducc/MediVoice-ct2) |
| VAD | Silero VAD | Built into faster-whisper |
| Diarization | Pyannote Speaker Diarization 3.1 | [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) |
| LLM | Llama 3.3 70B (via Groq) | [Groq Console](https://console.groq.com) |

## Setup Guide

First, clone the repository and enter the directory:
```bash
git clone https://github.com/lducc/MediVoice.git
cd MediVoice
```

### 1. Create Environment (Python 3.10)

**Option A: Using Conda**
```bash
conda create -n medivoice python=3.10 -c conda-forge -y
conda activate medivoice
```

**Option B: Using Venv**
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

---

### 2. Install Requirements

Install torch, torchvision and torchaudio:
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 
```
> **GPU Note:** The default command might install the CPU version of PyTorch. If you have an NVIDIA GPU that supports CUDA 12.4, run this instead:
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Install the remaining dependencies:
```bash
pip install -r requirements.txt
```

---

### 3. Setup Configure

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_huggingface_token_here
CORS_ORIGINS=*
HF_REPO_NAME=lducc/MediVoice-ct2
LOCAL_DIR=./model-ct2
```

#### Groq API Key (Required)
1. Go to [Groq Console](https://console.groq.com/keys)
2. Sign up or log in
3. Create a new API key
4. Copy and paste into `.env` file

#### HuggingFace Token (Optional - for Diarization)
1. Go to [HuggingFace](https://huggingface.co/)
2. Sign up or log in
3. Go to Settings → Access Tokens
4. Create a new token with read permissions
5. Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
6. Copy and paste into `.env` file

> **Note:** Without HF_TOKEN, the system will work but speaker diarization will be disabled.

---

### 4. Run with Docker (Alternative)

```bash
docker build -t medivoice .
docker run --gpus all -p 7860:7860 --env-file .env medivoice
```

## Usage

### Starting the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

### Web Interface (Gradio)

Open `http://localhost:8000` in your browser. You will see:

**Tab 1 — Speech to Text**
1. Upload an audio file (.wav, .mp3, .flac, .ogg)
2. Optionally check "Enable Speaker Diarization" to identify speakers
3. Click "Transcribe"
4. View the full transcript and timestamped segments

**Tab 2 — EMR Extraction**
1. Paste a transcript into the text box
2. Click "Extract Medical Data"
3. View the structured JSON with patient info, symptoms, medications, etc.

### REST API

The API is available alongside the Gradio UI for programmatic access.

API docs: `http://localhost:8000/docs`

#### 1. Health Check

**GET** `/ai/health`

```bash
curl http://localhost:8000/ai/health
```

Response:
```json
{
  "status": "ok",
  "asr": true,
  "diarization": true
}
```

#### 2. Speech-to-Text

**POST** `/ai/speech-to-text`

**Parameters:**
- `file`: Audio file (Supports: wav, mp3, flac, ogg, aiff)
- `enable_diarization`: Boolean (default: false)

**Without Diarization (Fast):**
```bash
curl -X POST "http://localhost:8000/ai/speech-to-text" \
  -F "file=@recording.wav" \
  -F "enable_diarization=false"
```

**With Diarization (Slower, identifies speakers):**
```bash
curl -X POST "http://localhost:8000/ai/speech-to-text" \
  -F "file=@recording.wav" \
  -F "enable_diarization=true"
```

#### 3. Generate EMR Draft

**POST** `/ai/generate-emr-draft`

**Request Body:**
```json
{
  "data": {
    "text": "Full transcript..."
  }
}
```

OR (with diarization):
```json
{
  "data": {
    "segments": [
      {
        "start": 0.5,
        "end": 3.2,
        "text": "transcript",
        "speaker": "SPEAKER_00"
      }
    ]
  }
}
```

## File Structure

```
MediVoice/
├── audio.py              # Audio loading utilities
├── configs.py            # Configuration and model download
├── engines.py            # Diarization and ASR engines
├── llm.py                # LLM-based EMR extraction
├── main.py               # FastAPI + Gradio application
├── Dockerfile            # Container deployment
├── .env                  # Environment variables (create this)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Performance Tips

### 1. GPU Acceleration
The system automatically uses CUDA if available. Ensure you have:
- NVIDIA GPU with CUDA support
- PyTorch with CUDA support

### 2. Batch Size
Adjust `BATCH_SIZE` in `configs.py` based on your GPU memory: 16 (default). 
Lower it if you get OOM errors. Use diarization only when you need to identify speakers.

## Troubleshooting

### Issue: "Diarization pipeline doesn't exist"
**Solution:** 
1. Add HF_TOKEN to `.env`
2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Restart the server

### Issue: CUDA out of memory
**Solution:**
1. Reduce `BATCH_SIZE` in `configs.py`
2. Use CPU instead by setting `COMPUTE_TYPE=int8` in `.env`
3. Process shorter audio segments

### Issue: Slow transcription on CPU
**Solution:**
CPU processing is 5-10x slower than GPU. Consider using a GPU instance or processing smaller audio files.
