# transcriber.py
"""Clean transcription handler using MLX Whisper"""
import numpy as np
import mlx_whisper
import time
from config import WHISPER_MODEL, SAMPLE_RATE, MIN_SEGMENT_DURATION


class RealtimeTranscriber:
    """Handles speech-to-text transcription"""
    
    def __init__(self, model_name: str = WHISPER_MODEL):
        print(f"Loading Whisper: {model_name}")
        self.model = model_name
        self.sample_rate = SAMPLE_RATE
        print("Whisper loaded")
    
    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text"""
        # Validate audio length
        duration = len(audio) / self.sample_rate
        if duration < MIN_SEGMENT_DURATION:
            print(f"Audio too short: {duration:.2f}s")
            return ""
        
        try:
            start_time = time.monotonic()
            
            # Run Whisper transcription
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model,
                verbose=False,
                language="en",
                fp16=False,
                temperature=0.0,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4
            )
            
            transcript = result["text"].strip()
            elapsed = time.monotonic() - start_time
            
            print(f"Transcribed in {elapsed:.2f}s (audio: {duration:.2f}s)")
            
            return transcript
        
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""