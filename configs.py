
import torch, os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv() 

#----------------- ENV CONFIG -----------------

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

HF_REPO_NAME = os.getenv("HF_REPO_NAME", "lducc/MediVoice")
LOCAL_DIR = Path(os.getenv("LOCAL_DIR", "./model-ct2"))

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
BATCH_SIZE = 16

# CTranslate2 compute type:
#   "int8"         → best for CPU (fast + small memory)
#   "float16"      → best for GPU with enough VRAM  
#   "int8_float16" → GPU with limited VRAM (int8 weights, float16 compute)
#   "auto"         → let CTranslate2 pick the best option
#
# This is set automatically based on your hardware, but you can override
# it via the COMPUTE_TYPE env var.
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8" if DEVICE == "cpu" else "int8_float16")


#----------------- DOWNLOAD MODEL -----------------

def download_model() -> str:
    if not LOCAL_DIR.exists():
        print(f"Download model from HuggingFace: {HF_REPO_NAME}")
        
        snapshot_download(
            repo_id=HF_REPO_NAME,
            local_dir=str(LOCAL_DIR),
            local_dir_use_symlinks=False,
            token=HF_TOKEN if HF_TOKEN else None
        )
    
    return str(LOCAL_DIR)
