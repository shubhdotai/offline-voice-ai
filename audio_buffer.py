# audio_buffer.py
"""Simplified audio buffer for speech segments"""
import numpy as np
import wave
from typing import Optional, List
from enum import Enum
from config import CHUNK_SIZE, SAMPLE_RATE, SAFETY_CHUNKS_BEFORE


class SpeechState(Enum):
    QUIET = "quiet"
    STARTING = "starting"
    SPEAKING = "speaking"
    STOPPING = "stopping"


class AudioBuffer:
    """Manages audio buffering with safety margins"""
    
    def __init__(self):
        self.pre_buffer: List[np.ndarray] = []
        self.active_segment: List[np.ndarray] = []
        self.is_capturing = False
    
    def add_chunk(self, chunk: np.ndarray, state: SpeechState):
        """Add chunk based on state"""
        chunk = chunk.copy()
        
        if state == SpeechState.QUIET:
            self.pre_buffer.append(chunk)
            if len(self.pre_buffer) > SAFETY_CHUNKS_BEFORE:
                self.pre_buffer.pop(0)
        
        elif state == SpeechState.STARTING:
            if not self.is_capturing:
                self.is_capturing = True
                self.active_segment = self.pre_buffer.copy()
            self.active_segment.append(chunk)
            self.pre_buffer.append(chunk)
            if len(self.pre_buffer) > SAFETY_CHUNKS_BEFORE:
                self.pre_buffer.pop(0)
        
        elif state == SpeechState.SPEAKING:
            self.active_segment.append(chunk)
        
        elif state == SpeechState.STOPPING:
            self.active_segment.append(chunk)
    
    def get_segment(self) -> Optional[np.ndarray]:
        if not self.active_segment:
            return None
        segment = np.concatenate(self.active_segment)
        self.active_segment = []
        self.is_capturing = False
        return segment


def save_audio_to_wav(audio: np.ndarray, filename: str, sample_rate: int = SAMPLE_RATE):
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(audio_int16.tobytes())


def split_audio_into_chunks(audio: np.ndarray, chunk_size: int = CHUNK_SIZE) -> List[np.ndarray]:
    num_chunks = len(audio) // chunk_size
    return [audio[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]
