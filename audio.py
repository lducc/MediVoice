import io
import torchaudio, torch


# Hardcoded supported extension: TODO: Add ffmpeg support

EXTENSIONS = {'.wav', '.flac', '.mp3', '.ogg'}

def is_supported_audio(filename: str) -> bool:
    import os
    ext = os.path.splitext(filename.lower())[1]
    return ext in EXTENSIONS

def load_audio(audio_bytes: bytes, target_sr=16000, filename: str = None):
    if not audio_bytes:
        raise ValueError("Audio bytes cannot be empty")
    
    if filename and not is_supported_audio(filename):
        raise ValueError(
            f"Unsupported audio format. Supported: {', '.join(sorted(EXTENSIONS))}"
        )
    
    try:
        # torchaudio.load returns (Tensor[channels, samples], sample_rate)
        wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
        
        #Convert audio to mono if stereo/ multichannel
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        
        #Resample to 16k if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            wav = resampler(wav)
        
        # Squeeze removes the channel dimension: (1, samples) -> (samples,)
        audio_array = wav.squeeze().numpy()
        
        return audio_array, target_sr
        
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {str(e)}")

#----------------- CHUNKING -----------------
def merge_segments_to_chunks(segments, sr=16000, max_duration_sec=30.0):
    if not segments: 
        return []
    
    max_samples = int(max_duration_sec * sr)
    merged = []
    current_chunk = segments[0].copy()
    
    #Combine segments to chunks < 30s 
    for next_seg in segments[1:]:
        current_dur = current_chunk['end'] - current_chunk['start']
        next_dur = next_seg['end'] - next_seg['start']
        gap = next_seg['start'] - current_chunk['end']
        
        if (current_dur + gap + next_dur) < max_samples:
            current_chunk['end'] = next_seg['end']
        else:
            merged.append(current_chunk)
            current_chunk = next_seg.copy()
            
    merged.append(current_chunk)
    return merged

def merge_segments_by_speaker(segments, sr=16000, max_duration_sec=30.0):
    #Merge consecutive segments from the same speaker (diarization)
    if not segments:
        return []
    
    max_samples = int(max_duration_sec * sr)
    chunks = []
    
    sorted_segments = sorted(segments, key=lambda x: x['start'])
    
    current_chunk = {
        'start': sorted_segments[0]['start'],
        'end': sorted_segments[0]['end'],
        'speaker': sorted_segments[0]['speaker']
    }
    
    for seg in sorted_segments[1:]:
        current_dur = current_chunk['end'] - current_chunk['start']
        gap = seg['start'] - current_chunk['end']
        next_dur = seg['end'] - seg['start']
        
        # Merge when it is the same speaker
        same_speaker = seg['speaker'] == current_chunk['speaker']
        gap_acceptable = gap < (2.0 * sr)  # Less than 2 seconds pause -> add gap, else -> new chunk
        duration_ok = (current_dur + gap + next_dur) < max_samples
        
        if same_speaker and gap_acceptable and duration_ok:
            current_chunk['end'] = seg['end']
        else:
            chunks.append(current_chunk)
            current_chunk = {
                'start': seg['start'],
                'end': seg['end'],
                'speaker': seg['speaker']
            }
    
    chunks.append(current_chunk)
    
    return chunks