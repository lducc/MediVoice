import os, sys, torch, json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from huggingface_hub import snapshot_download   #Easy way to download hugging face models 
from typing import List, Tuple, Optional, Dict, Any
import shutil, librosa
import librosa, uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from openai import OpenAI

# -------------------------- CONFIGS --------------------------------------
load_dotenv() 

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
REPO_ID = "leduckhai/MultiMed-ST"
SUBFOLDER = "asr/whisper-small-vietnamese"
LOCAL_DIR = "./whisper_model_local"
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

model: Optional[WhisperForConditionalGeneration] = None
processor: Optional[WhisperProcessor] = None
device: str = "cpu"
origins = [o.strip() for o in CORS_ORIGINS.split(",")] if CORS_ORIGINS else ["*"]

# OpenAI client
client = OpenAI(api_key = OPENAI_KEY)


## -------------------------- ASR MODEL --------------------------------------

def load_model() -> Tuple[Any, Any, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = snapshot_download(repo_id = REPO_ID,
                            allow_patterns = f"{SUBFOLDER}/*",
                            ignore_patterns = ["*runs/*", "*.tfevents*"],  #github has some log files we dont need
                            local_dir = LOCAL_DIR)
                            # local_dir_use_symlinks = False)

    config_path = os.path.join(root, SUBFOLDER)
    weight_path = os.path.join(config_path, "checkpoint-5000")

    #Download the config and weight files
    if os.path.exists(config_path) and os.path.exists(weight_path):
        for filename in os.listdir(config_path):
            if filename.endswith(".json") or filename.endswith(".txt"):
                src = os.path.join(config_path, filename)
                dst = os.path.join(weight_path, filename)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)

    processor = WhisperProcessor.from_pretrained(config_path)
    model = WhisperForConditionalGeneration.from_pretrained(weight_path, 
                                                            use_safetensors = True,
                                                            low_cpu_mem_usage = True).to(device)
    model.eval()

    return model, processor, device
    
def transcribe_audio(audio_path: str) -> str:   
    #Model is trained on 16000khz audio so the audio must match it
    speech, _ = librosa.load(audio_path, sr = 16000)
    
    tensors = processor(speech, sampling_rate = 16000, return_tensors = "pt") #  <--- Converts audio to tensors
    features = tensors.input_features.to(device)

    with torch.no_grad():
        ids = model.generate(features, language = "vi", task = "transcribe") # Makes tokens ig for decoding later

    text = processor.batch_decode(ids, skip_special_tokens=True)[0] #special token is html tags kinda -> trash token dont need it
    return text  


## -------------------------- TEXT EXTRACTION (OpenAI) --------------------------------------

def extract_medical_data(transcript: str, lang: str) -> Dict[str, Any]:    
    system_prompt = """
    Bạn là một nhân viên y khoa AI chuyên nghiệp tên là MediFlow.
    Nhiệm vụ của bạn là phân tích bản gỡ băng hội thoại khám bệnh và trích xuất thông tin y tế có cấu trúc.
    
    Kết quả trả về BẮT BUỘC phải là định dạng JSON hợp lệ (không có markdown block), tuân theo cấu trúc chính xác sau:
    {
      "patient_info": { 
          "age": int or null, 
          "gender": "Nam"/"Nữ" or null, 
          "nationality": str or null 
      },
      "chief_complaint": str (Lý do chính đi khám),
      "hpi": { 
        "duration": str hoặc null (ví dụ: "3 ngày nay"), 
        "symptoms": List[str] (các triệu chứng tích cực), 
        "negative_symptoms": List[str] (các triệu chứng âm tính/không có), 
        "description": str (Tóm tắt bệnh sử ngắn gọn) 
      },
      "past_medical_history": { 
          "chronic_diseases": List[str] (Bệnh mãn tính), 
          "allergies": List[str] (Dị ứng), 
          "current_medications": List[str] (Thuốc đang dùng) 
      },
      "assessment": List[str] (Chẩn đoán sơ bộ hoặc xác định),
      "plan": { 
          "tests": List[str] (Chỉ định cận lâm sàng), 
          "medications": List[str] (Thuốc kê đơn), 
          "advice": List[str] (Lời khuyên/Dặn dò) 
      },
      "missing_fields": List[str] (Liệt kê các thông tin quan trọng còn thiếu, ví dụ: "Tuổi", "Tiền sử dị ứng", "Huyết áp"),
      "confidence": float (0.0 đến 1.0 dựa trên độ rõ ràng của thông tin)
    }
    
    Yêu cầu:
    1. Nếu thông tin không có trong văn bản, hãy dùng null hoặc mảng rỗng [].
    2. Sử dụng thuật ngữ y khoa Tiếng Việt chuẩn.
    3. Trích xuất trung thực, không tự bịa đặt thông tin không có trong văn bản.
    """

    user_prompt = f"Transcript: {transcript}"

    try:
        response = client.chat.completions.create(
            model = "gpt-4o-mini", 
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format = {"type": "json_object"}, 
            temperature = 0.1 # Low temperature for consistent extraction
        )
        
        json_content = json.loads(response.choices[0].message.content)
        
        # Metadata
        json_content["transcript"] = transcript
        json_content["lang"] = lang
        
        return json_content

    except Exception as e:
        print(f"OpenAI Error: {e}")
        return {
            "transcript": transcript,
            "error": str(e)
        }
    
## -------------------------- FASTAPI --------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, device
    model, processor, device = load_model() #Loads the model as soon as the server is running
    yield
    if torch.cuda.is_available():   
        torch.cuda.empty_cache()

app = FastAPI(lifespan = lifespan)
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True,
                    allow_methods=["*"], allow_headers=["*"])   #Middleware allows backend to talk to the server

@app.get("/health") #GET: re3ad-only, diagnostic check
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": device,
        "openai_configured": bool(OPENAI_KEY),
    }

@app.post("/analyze") #POST: sends the audio data to the server -> asr and openai extracts the data -> json
async def analyze_audio(file: UploadFile = File(...), lang: str = "vi"):
    temp_filename = f"temp_{file.filename}"     # Save temporary file for audio reading
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        transcript = transcribe_audio(temp_filename)
        # print(f"Transscript: {transcript}")
        response_json = extract_medical_data(transcript, lang)

        return response_json

    except Exception as e:
        return
    
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8000)