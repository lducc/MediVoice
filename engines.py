import os, torch
import numpy as np
import functools
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pyannote.audio import Pipeline
import configs

#--------------------VAD ENGINE----------------------
#Uses VAD (Silero) if enable_diarization = False for voice segmentation

class VAD:
    def __init__(self):
        self.model = None
        self.utils = None
        
    def load(self):        
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False, 
            trust_repo=True
        )
        self.utils = utils
        self.model.to(configs.DEVICE)
        

    def get_segments(self, audio_array: np.ndarray, sr = 16000):
        #Get segments from audio array 
        if not self.model: self.load()
        
        (get_speech_timestamps, _, _, _, _) = self.utils
        wav = torch.from_numpy(audio_array).float().to(configs.DEVICE)
        
        speech_timestamps = get_speech_timestamps(
            wav, 
            self.model,
            sampling_rate=sr,
            threshold=0.2,
            min_speech_duration_ms=500,
            min_silence_duration_ms=500,
            speech_pad_ms=500
        )
        
        return speech_timestamps

#--------------------DIARIZATION ENGINE----------------------
#Uses Pyannote if enable_diarization = True
class DiarizationEngine:
    def __init__(self):
        self.pipeline = None
    
    def load(self):
        if not configs.HF_TOKEN:
            print("HF_TOKEN doesn't exist. Diarization disabled.")
            return
                
        #Force the huggingface model to run using weights_only = False to load it 
        #(torch 2.6+ automatically sets weights_only = True)
        original_load = torch.load

        @functools.wraps(original_load)
        def trusted_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)

        try:
            torch.load = trusted_load
            
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=configs.HF_TOKEN
            )
            
            if self.pipeline:
                self.pipeline.to(torch.device(configs.DEVICE))

        except Exception as e:
            print(f"Failed to load diarization pipeline: {e}")
            self.pipeline = None
            
        finally:
            torch.load = original_load
    
    def get_speaker_segments(self, audio_array: np.ndarray, sr=16000, min_speakers=1, max_speakers=2):
        #Full diarization (segmentation + speaker labeling), returns segments with speaker labels in samples
    
        if not self.pipeline:
            print("Diarization pipeline doesn't exist")
            return []
        
        # Prepare audio for Pyannote
        waveform = torch.from_numpy(audio_array).float().unsqueeze(0)
        audio = {"waveform": waveform, "sample_rate": sr}
        
        # Run diarization
        diarization = self.pipeline(
            audio,
            min_speakers = min_speakers,
            max_speakers = max_speakers,
        )
        
        #Convert it all into segments with timestamps
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
        
        # unique_speakers = set(s['speaker'] for s in segments)
        # print(f"{len(unique_speakers)} speaker(s): {', '.join(sorted(unique_speakers))}")
        # print(f"Total: {len(segments)}")
        
        # for i, seg in enumerate(segments):
        #     start_sec = seg['start'] / sr
        #     end_sec = seg['end'] / sr
        #     duration = end_sec - start_sec
        #     print(f"[{i}] {seg['speaker']}:{start_sec:.2f}s - {end_sec:.2f}s ({duration:.2f}s)")
        
        
        # speaker_counts = {}
        # for seg in segments:
        #     speaker_counts[seg['speaker']] = speaker_counts.get(seg['speaker'], 0) +1

        
        return segments

#--------------------ASR ENGINE----------------------
class ASR:
    def __init__(self):
        self.model = None
        self.processor = None

    def load(self):
        # Download model if needed
        model_dir = configs.download_model()
        
        self.processor = WhisperProcessor.from_pretrained(model_dir)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_dir, 
            local_files_only=True,
            torch_dtype=configs.TORCH_DTYPE,
            low_cpu_mem_usage=True
        ).to(configs.DEVICE)

        self.model.eval()
        
    def transcribe_chunks(self, audio_array, chunks):        
        if not self.model: 
            self.load()
        
        input_features = []
        valid_chunks = [] 

        for chunk in chunks:
            start_sample = chunk['start']
            end_sample = chunk['end']
            audio_crop = audio_array[start_sample: end_sample]
            
            if len(audio_crop) < 1600: continue 
            
            inputs = self.processor(audio_crop, sampling_rate=16000, return_tensors="pt")
            input_features.append(inputs.input_features)
            valid_chunks.append(chunk)

        all_decoded_texts = []
        total_chunks = len(input_features)
                
        for i in range(0, total_chunks, configs.BATCH_SIZE):
            batch_slice = input_features[i: i + configs.BATCH_SIZE]
            batch_tensor = torch.cat(batch_slice, dim=0).to(configs.DEVICE, dtype=configs.TORCH_DTYPE)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    batch_tensor, 
                    language="vi", 
                    task="transcribe",
                    attention_mask=torch.ones_like(batch_tensor[:, 0, :]).long() #<-- Removes the warnings
                )
            
            batch_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            all_decoded_texts.extend(batch_text)
            
            #Clear cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        final_segments = []
        for i, text in enumerate(all_decoded_texts):
            text = text.strip()
            if not text: continue
            
            chunk = valid_chunks[i]
            segment = {
                "start": chunk['start'] / 16000,
                "end": chunk['end'] / 16000,
                "text": text,
                "speaker": chunk.get('speaker', 'UNKNOWN')
            }
            final_segments.append(segment)
            
            # duration = segment['end'] - segment['start']
            # print(f"[{segment['speaker']}] {segment['start']:.2f}s - {segment['end']:.2f}s ({duration:.2f}s)")
            # print(f" {segment['text']}")
            
        return final_segments

