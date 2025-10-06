import numpy as np
import mlx_whisper
import time


class RealtimeTranscriber:
    def __init__(self, model_name: str = "mlx-community/whisper-small.en-mlx-q4", sample_rate: int = 16000):
        print(f"Loading Whisper model: {model_name}")
        self.model = model_name
        self.sample_rate = sample_rate
        
    def transcribe(self, audio_data: np.ndarray) -> str:
        if len(audio_data) < self.sample_rate * 0.3:
            print("Audio segment too short for transcription.")
            return ""
        
        try:
            start_time = time.monotonic()
            result = mlx_whisper.transcribe(
                audio_data,
                path_or_hf_repo=self.model,
                verbose=False,
                language="en",
                fp16=False,
                temperature=0.0,
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
            )
            transcript = result["text"].strip()
            elapsed = time.monotonic() - start_time
            print(f"Transcription took {elapsed:.2f}s | Length: {len(audio_data)/self.sample_rate:.2f}s")
            return transcript
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""