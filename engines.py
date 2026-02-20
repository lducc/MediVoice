import os, torch
import numpy as np
import functools
import warnings
from faster_whisper import WhisperModel, BatchedInferencePipeline

#Monkey patching huggingface_hub to fix the issue with pyannote
import huggingface_hub
_original_hf_download = huggingface_hub.hf_hub_download
@functools.wraps(_original_hf_download)
def _patched_hf_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    return _original_hf_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _patched_hf_download

from pyannote.audio import Pipeline
import configs

# Suppress warnings
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*std().*degrees of freedom.*")
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")


#--------------------DIARIZATION ENGINE----------------------
class DiarizationEngine:
    def __init__(self):
        self.pipeline = None
    
    def load(self):
        if not configs.HF_TOKEN:
            print("HF_TOKEN doesn't exist. Diarization disabled.")
            return

        original_load = torch.load
        @functools.wraps(original_load)
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)

        try:
            torch.load = patched_load
            os.environ["HF_TOKEN"] = configs.HF_TOKEN
            
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
            )
            
            if self.pipeline:
                self.pipeline.to(torch.device(configs.DEVICE))

        except Exception as e:
            print(f"Failed to load diarization pipeline: {e}")
            self.pipeline = None
            
        finally:
            torch.load = original_load
    
    def get_speaker_segments(self, audio_array: np.ndarray, sr: int = 16000, min_speakers: int = 1, max_speakers: int = 2) -> list[dict]:
        if not self.pipeline:
            print("Diarization pipeline doesn't exist")
            return []
        
        waveform = torch.from_numpy(audio_array).float().unsqueeze(0)
        audio = {"waveform": waveform, "sample_rate": sr}
        
        diarization = self.pipeline(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                'start': int(turn.start * sr),  
                'end': int(turn.end * sr),
                'speaker': speaker
            })
        
        if not segments:
            print("No speakers detected by Pyannote")
            return []
        
        return segments


#--------------------ASR ENGINE----------------------
class ASR:
    def __init__(self):
        self.model = None
        self.batched_model = None

    def load(self):
        model_dir = configs.download_model()
        
        self.model = WhisperModel(
            model_dir,
            device=configs.DEVICE,
            compute_type=configs.COMPUTE_TYPE,
        )
        
        self.batched_model = BatchedInferencePipeline(model=self.model)
        
        print(f"ASR loaded: device={configs.DEVICE}, compute_type={configs.COMPUTE_TYPE}")

    def transcribe(self, audio_array: np.ndarray) -> list[dict]:
        if not self.model: 
            self.load()
        
        segments, info = self.batched_model.transcribe(
            audio_array,
            language="vi",
            beam_size=1,                       
            batch_size=configs.BATCH_SIZE,
            without_timestamps=True,             
            condition_on_previous_text=False,     
            vad_filter=True,
            vad_parameters={
                "threshold": 0.3,
                "min_speech_duration_ms": 500,
                "min_silence_duration_ms": 500,
            },
        )
        
        results = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                results.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": text,
                    "speaker": "UNKNOWN"
                })
        
        return results

    def transcribe_with_speakers(self, audio_array: np.ndarray, speaker_segments: list[dict], sr: int = 16000) -> list[dict]:
        #Transcribe full audio -> Assign speakers from diarization.
        
        text_segments = self.transcribe(audio_array)
        
        for seg in text_segments:
            seg_start = seg["start"] * sr 
            seg_end = seg["end"] * sr
            
            best_speaker = "UNKNOWN"
            best_overlap = 0
            
            for spk_seg in speaker_segments:
                overlap_start = max(seg_start, spk_seg["start"])
                overlap_end = min(seg_end, spk_seg["end"])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = spk_seg["speaker"]
            
            seg["speaker"] = best_speaker
        
        return text_segments
