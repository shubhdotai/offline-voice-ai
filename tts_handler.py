# tts_handler.py
"""Clean TTS handler using Kokoro"""
import torch
import numpy as np
import io
import wave
from kokoro import KPipeline
from config import TTS_MODEL, TTS_VOICE, TTS_SPEED, TTS_SAMPLE_RATE


class TTSHandler:
    """Handles text-to-speech generation"""
    
    def __init__(
        self,
        repo_id: str = TTS_MODEL,
        voice: str = TTS_VOICE,
        speed: float = TTS_SPEED,
        sample_rate: int = TTS_SAMPLE_RATE
    ):
        print(f"Loading TTS: {repo_id}")
        self.pipeline = KPipeline(lang_code='a', repo_id=repo_id)
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        print("TTS loaded")
    
    def generate_speech(self, text: str) -> bytes:
        """Generate speech audio from text"""
        if not text or not text.strip():
            return b''
        
        try:
            # Generate audio chunks
            audio_chunks = []
            for result in self.pipeline(text, voice=self.voice, speed=self.speed):
                if result.audio is not None:
                    audio_chunks.append(result.audio)
            
            if not audio_chunks:
                print("No audio generated")
                return b''
            
            # Concatenate and convert to WAV
            full_audio = torch.cat(audio_chunks, dim=0)
            audio_array = full_audio.numpy()
            wav_bytes = self._to_wav_bytes(audio_array)
            
            duration = len(audio_array) / self.sample_rate
            print(f"Generated {duration:.2f}s audio")
            
            return wav_bytes
        
        except Exception as e:
            print(f"TTS error: {e}")
            return b''
    
    def _to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert audio array to WAV bytes"""
        # Clip and convert to int16
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Write to WAV buffer
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()