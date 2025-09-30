import numpy as np
import mlx_whisper
from typing import Optional
import time


class RealtimeTranscriber:
    def __init__(
        self,
        model_name: str = "mlx-community/whisper-small.en-mlx-q4",
        sample_rate: int = 16000,
    ):
        """Real-time transcriber using MLX Whisper with optimizations."""
        print(f"Loading Whisper model: {model_name}")
        self.model = model_name
        self.sample_rate = sample_rate
        
    def transcribe(self, audio_data: np.ndarray, initial_prompt: str = "") -> str:
        """
        Transcribe audio segment with contextual awareness.
        
        Args:
            audio_data: Audio samples as numpy array
            initial_prompt: Previous transcript for context continuity
            
        Returns:
            Transcribed text string
        """
        start_time = time.monotonic()
        
        # Minimum audio length check
        if len(audio_data) < self.sample_rate * 0.3:
            print("Audio segment too short for transcription.")
            return ""
        
        try:
            # Transcribe with context
            result = mlx_whisper.transcribe(
                audio_data,
                path_or_hf_repo=self.model,
                verbose=False,
                language="en",
                fp16=False,
                initial_prompt=initial_prompt,  # Context from previous transcript
                condition_on_previous_text=True  # Enable contextual awareness
            )
            
            end_time = time.monotonic()
            transcript = result["text"].strip()
            
            print(f"Transcription took {end_time - start_time:.2f}s | Length: {len(audio_data)/self.sample_rate:.2f}s")
            
            return transcript
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""