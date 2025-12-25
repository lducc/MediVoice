# MediVoice

A Speech-to-Text model for medical Vietnamese audio, currently configured with FastAPI and OpenAI for language processing.

## Files

*   `main.py` – The script to run inference on audio files
*   `requirements.txt` – Needed library dependencies
*   `README.md` – This file you are reading
*   `sample.wav` - A sample audio for testing
*   `.env` – Configuration file for API Keys

## Setup Guide

First, clone the repository and enter the directory:

```bash
git clone https://github.com/your-username/MediFlow.git
cd MediFlow
```

### 1. Create Environment (Python 3.10.19)

**Option A: Using Conda**

```bash
conda create -n medivoice python=3.10
conda activate medivoice
```

**Option B: Using Venv**

```bash
python -m venv venv
# Windows:
    venv\Scripts\activate
# Mac/Linux:
    source venv/bin/activate
```

---

### 2. Install Requirements

> **GPU Note:**
> The default command might install the CPU version of PyTorch. If you have an NVIDIA GPU (RTX 3050, etc.), run this **before** installing requirements:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

Install the dependencies:
```bash
pip install -r requirements.txt
```

---

### 3. Configure API Keys

Create a file named `.env` in the project root and add your OpenAI Key:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
```

---

## How to Run

Start the server:

```bash
python main.py
```

The server will start at: `http://localhost:8000`

### How to Test (Swagger UI)

1.  Open your browser to: **[http://localhost:8000/docs](http://localhost:8000/docs)**
2.  Click on the green **`POST /analyze`** bar.
3.  Click **Try it out**.
4.  Upload `sample.wav` in the **file** field.
5.  Click **Execute**.

*Note: The first run will automatically download the ~1GB model. Subsequent runs will be instant.*

