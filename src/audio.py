import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import librosa

TARGET_SR = 16000


def load_audio(audio_bytes: Optional[bytes] = None, *, path: Optional[str] = None, filename: Optional[str] = None) -> Tuple[np.ndarray, int]:
    #Load audio → mono float32 numpy array at 16kHz
    
    if not path and not audio_bytes:
        raise ValueError("Provide either path or audio_bytes")

    if not path:
        ext = os.path.splitext(filename)[1] if filename else '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(audio_bytes)
            path = f.name

    audio_np, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio_np, TARGET_SR
