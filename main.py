import os, time, uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import torch, gc
import configs
import audio as audio_utils
from engines import ASR, DiarizationEngine
from llm import extract_medical_data

asr_engine = ASR()
diarization_engine = DiarizationEngine()

#----------------- PIPELINE -----------------
def run_pipeline(audio_bytes: bytes, enable_diarization: bool = False):
    audio, sr = audio_utils.load_audio(audio_bytes) 

    if enable_diarization and diarization_engine.pipeline:        
        # Diarization mode: get speaker segments -> transcribe + align
        segments = diarization_engine.get_speaker_segments(audio, sr)
        
        if not segments:
            return {"text": "", "segments": []}
        
        # Transcribe full audio -> assign speakers by overlap
        segment_results = asr_engine.transcribe_with_speakers(audio, segments, sr)
        
    else:        
        segment_results = asr_engine.transcribe(audio)
    
    full_text = " ".join([seg["text"] for seg in segment_results])
    
    return {
        "text": full_text,
        "segments": segment_results
    }

#----------------- FASTAPI -----------------

origins = [o.strip() for o in configs.CORS_ORIGINS.split(",")] if configs.CORS_ORIGINS else ["*"]

def flush_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@asynccontextmanager
async def lifespan(app: FastAPI):
    asr_engine.load()
    diarization_engine.load()
    yield
    flush_memory()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.get("/ai/health")
def health():
    return {
        "status": "ok", 
        "asr": asr_engine.model is not None,
        "diarization": diarization_engine.pipeline is not None,
    }

@app.post("/ai/speech-to-text")
async def speech_to_text(file: UploadFile = File(...), enable_diarization: bool = False):
    """
    Transcribe audio to text (Optional: Diarization for speaker labeling)
    
    - enable_diarization=False: VAD only (fast, no speakers)
    - enable_diarization=True: Full diarization (slower, with speakers)
    """
    try:        
        if file.filename and not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aiff')):
            raise HTTPException(400, f"Unsupported audio format: {file.filename}")
        
        content = await file.read()
        
        t0 = time.time()
        result = run_pipeline(content, enable_diarization)
        duration = time.time() - t0
        
        print(f"Finished in {duration:.2f}s")
        
        return {"data": result}
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))

@app.post("/ai/generate-emr-draft")
async def generate_emr(payload: dict = Body(...)):
    """
    Extract JSON medical data from transcript or conversation segments.
    Accepts either:
    - {"data": {"text": "..."}} for plain transcript (VAD mode)
    - {"data": {"segments": [...]}} for diarized conversation
    """
    data = payload.get("data", {})
    
    text = data.get("text")
    segments = data.get("segments")
    
    if not text and not segments:
        raise HTTPException(400, "Missing text or segments in payload")
    
    return {
        "status": "success", 
        "emr": extract_medical_data(transcript=text, segments=segments)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)