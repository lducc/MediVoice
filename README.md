# MediVoice

A Speech-to-Text model for medical Vietnamese audio.

## Files

*   `main.py` – The script to run inference on audio files
*   `requirements.txt` – Needed library dependencies
*   `README.md` – This file you are reading
*   `sample.wav` - A sample audio for testing


## Setup Guide

First, clone the repository and enter the directory:

```bash
git clone https://github.com/your-username/MediFlow.git
cd MediFlow
```

### 1. Create an Environment (Choose one)

**Option A: Using Conda**

```bash
conda create -n <env-name> python=3.10 -y
conda activate <env-name>
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

Install the dependencies:
```bash
pip install -r requirements.txt
```
> *Note:*
> The default `pip install -r requirements.txt` might install the CPU version of PyTorch. If you have an NVIDIA GPU that supports CUDA, run this:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

## How to Run

Run the script through the cmd (**Must have 2 parameters: the main.py file + the audio file you choose earlier**):

**Option 1: File in the same folder**

```bash
python main.py sample.wav
```

**Option 2: Full file path**

```bash
python main.py "C:\Users\Doctor\Documents\Recordings\patient_01.wav"
```

*Note:* The first time you run this, it will automatically download the ~1GB model. Runs after it will be instant.



