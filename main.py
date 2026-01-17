import time, uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import torch, gc
import configs
import audio as audio_utils
from engines import VAD, ASR, DiarizationEngine
from llm import extract_medical_data

vad_engine = VAD()
asr_engine = ASR()
diarization_engine = DiarizationEngine()

#----------------- PIPELINE -----------------

def run_pipeline(audio_bytes: bytes, enable_diarization: bool = False):
    audio, sr = audio_utils.load_audio(audio_bytes) 

    if enable_diarization and diarization_engine.pipeline:        
        #Pyannote for handling both segmentation and speaker labeling
        segments = diarization_engine.get_speaker_segments(audio, sr)
        
        if not segments:
            return {"text": "", "segments": []}
        
        chunks = audio_utils.merge_segments_by_speaker(segments, sr)
        
    else:        
        #Voice activity detection using Silero, no speakers diarization
        vad_segments = vad_engine.get_segments(audio)
        
        if not vad_segments:
            return {"text": "", "segments": []}
        
        # print(f"{len(vad_segments)} speech segments")
        
        chunks = audio_utils.merge_segments_to_chunks(vad_segments, sr)
        # print(f"{len(chunks)} VAD chunks")
    
    # for i, chunk in enumerate(chunks):
    #     speaker = chunk.get('speaker', 'UNKNOWN')
    #     start_sec = chunk['start'] / sr
    #     end_sec = chunk['end'] / sr
    #     duration = end_sec - start_sec
        # print(f"[{i}] {speaker}: {start_sec:.2f}s - {end_sec:.2f}s ({duration:.2f}s)")
    
    segment_results = asr_engine.transcribe_chunks(audio, chunks)
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
    vad_engine.load()
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
        "vad": vad_engine.model is not None, 
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
    
    # Pass segments if available (diarized), otherwise use text (VAD-only)
    return {
        "status": "success", 
        "emr": extract_medical_data(transcript=text, segments=segments)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)