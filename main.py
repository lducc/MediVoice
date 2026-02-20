import os, time, json, uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import torch, gc
import gradio as gr
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

#----------------- GRADIO UI -----------------

def gradio_transcribe(audio_path, enable_diarization):
    """Gradio handler for speech-to-text."""
    if audio_path is None:
        return "Please upload an audio file.", ""
    
    t0 = time.time()
    audio, sr = audio_utils.load_audio(path=audio_path)
    
    if enable_diarization and diarization_engine.pipeline:
        segments = diarization_engine.get_speaker_segments(audio, sr)
        if not segments:
            return "No speech detected.", ""
        segment_results = asr_engine.transcribe_with_speakers(audio, segments, sr)
    else:
        segment_results = asr_engine.transcribe(audio)
    
    duration = time.time() - t0
    full_text = " ".join([seg["text"] for seg in segment_results])
    
    segments_display = ""
    for seg in segment_results:
        speaker = seg.get("speaker", "")
        prefix = f"[{speaker}] " if speaker else ""
        segments_display += f"{prefix}[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}\n"
    
    segments_display += f"\n--- Processed in {duration:.2f}s ---"
    return full_text, segments_display


def gradio_extract_emr(transcript):
    """Gradio handler for EMR extraction."""
    if not transcript or not transcript.strip():
        return "Please provide a transcript first."
    
    from llm import extract_medical_data
    result = extract_medical_data(transcript=transcript)
    return json.dumps(result, ensure_ascii=False, indent=2)


with gr.Blocks(title="MediVoice", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🩺 MediVoice\nVietnamese medical speech-to-text and EMR extraction")
    
    with gr.Tab("Speech to Text"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Upload Audio")
                diarize_toggle = gr.Checkbox(label="Enable Speaker Diarization", value=False)
                transcribe_btn = gr.Button("Transcribe", variant="primary")
            with gr.Column():
                text_output = gr.Textbox(label="Full Transcript", lines=6)
                segments_output = gr.Textbox(label="Segments", lines=10)
        
        transcribe_btn.click(
            fn=gradio_transcribe,
            inputs=[audio_input, diarize_toggle],
            outputs=[text_output, segments_output]
        )
    
    with gr.Tab("EMR Extraction"):
        with gr.Row():
            with gr.Column():
                emr_input = gr.Textbox(label="Paste Transcript", lines=8, placeholder="Paste transcript here...")
                emr_btn = gr.Button("Extract Medical Data", variant="primary")
            with gr.Column():
                emr_output = gr.JSON(label="EMR Data")
        
        emr_btn.click(
            fn=gradio_extract_emr,
            inputs=[emr_input],
            outputs=[emr_output]
        )

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)