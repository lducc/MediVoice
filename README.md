# MediVoice API

MediVoice is a medical voice transcription system with speaker diarization and LLM-assisted Electronic Medical Record (EMR) extraction. It converts doctor-patient conversations into structured medical data using state-of-the-art AI models.

## Features

- **Audio Transcription**: Converts audio to text using Whisper (Vietnamese language)
- **Speaker Diarization**: Identifies and separates different speakers (doctor vs patient)
- **EMR Extraction**: Automatically extracts structured medical data from conversations using LLM
- **Fast Processing**: GPU-accelerated with batch processing
- **Confidence Scores**: Provides confidence metrics based on log-probs for extracted data


##  Setup Guide

First, clone the repository and enter the directory:
```bash
git clone https://github.com/lducc/MediVoice.git
cd MediVoice
```
### 1. Create Environment (Python 3.10.19)

[](https://github.com/lducc/MediVoice#1-create-environment-python-31019)


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


> **GPU Note:**  The default command might install the CPU version of PyTorch. If you have an NVIDIA GPU (RTX 3050, etc.), run this  **before**  installing requirements:
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
Install the dependencies:
```bash
pip install -r requirements.txt
```
---
### 3. Setup Configure

Create a `.env` file in the project root:
```env
HF_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_key_here
CORS_ORIGINS=*
HF_REPO_NAME=lducc/MediVoice
LOCAL_DIR=./MediVoice
```


#### OpenAI API Key (Required)
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy and paste into `.env` file

#### HuggingFace Token (Optional - for Diarization)
1. Go to [HuggingFace](https://huggingface.co/)
2. Sign up or log in
3. Go to Settings → Access Tokens
4. Create a new token with read permissions
5. Accept the terms for [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
6. Copy and paste into `.env` file

>**Note:** Without HF_TOKEN, the system will work but speaker diarization will be disabled.

## File Structure

```
MediVoice/
├── audio.py                    # Audio processing and chunking
├── configs.py                  # Configuration and model download
├── engines.py                  # VAD, Diarization, and ASR engines
├── llm.py                      # LLM-based EMR extraction
├── main.py                     # FastAPI application
├── .env                        # Environment variables (create this)
├── requirements.txt            # Python dependencies (create this)
└── README.md                   # This file
```

## Usage

### Starting the Server

```bash
python main.py
```

The server will start on `http://localhost:8000`

### API Endpoints

#### 1. Health Check

**GET** `/health`

Check if all models are loaded correctly.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "vad": true,
  "asr": true,
  "diarization": true
}
```

#### 2. Speech-to-Text

**POST** `/ai/speech-to-text`

Transcribe audio files to text with optional speaker diarization.

**Parameters:**
- `file`: Audio file (Supports: wav, mp3, flac, ogg) *(TODO: Add ffmpeg for other extensions support)*

 `enable_diarization`: Boolean (default: false)

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

Extract structured medical data from transcription.

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

## Performance Tips

### 1. GPU Acceleration
The system automatically uses CUDA if available. Ensure you have:
- NVIDIA GPU with CUDA support
- PyTorch with CUDA support

### 2. Batch Size
Adjust `BATCH_SIZE` in `configs.py` based on your GPU memory: 16 (default). 
You could and should lower it down if OOM. Use diarization only when you need to identify speakers.

## Troubleshooting

### Issue: "OPENAI_API_KEY is required"
**Solution:** Add your OpenAI API key to the `.env` file

### Issue: "Diarization pipeline doesn't exist"
**Solution:** 
1. Add HF_TOKEN to `.env`
2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Restart the server

### Issue: CUDA out of memory
**Solution:**
1. Reduce `BATCH_SIZE` in `configs.py`
2. Use CPU instead by setting `DEVICE = "cpu"` in `configs.py`
3. Process shorter audio segments

### Issue: Slow transcription on CPU
**Solution:**
This is expected. CPU processing is 5-10x slower than GPU. Consider:
1. Using a GPU instance
2. Processing smaller audio files

## Model Information

- **ASR Model**: Whisper (Vietnamese fine-tuned)
- **VAD Model**: Silero VAD
- **Diarization**: Pyannote Speaker Diarization 3.1
- **LLM**: GPT-4o-mini (for EMR extraction)

