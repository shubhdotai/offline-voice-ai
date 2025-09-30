import json
import base64
import numpy as np
import onnxruntime as ort
from torch import threshold
from transformers import WhisperFeatureExtractor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import wave
from datetime import datetime
from enum import Enum
from typing import Optional, List

# Configuration
ALPHA = 0.1
THRESHOLD = 0.5
START_DELTA = 0.2
STOP_DELTA = 0.3
FALLBACK_THRESHOLD = 0.1

class SpeechState(Enum):
    QUIET = "quiet"
    STARTING = "starting"
    SPEAKING = "speaking"
    STOPPING = "stopping"

class SpeechSegment:
    def __init__(self, start_time: float):
        self.start_time = start_time
        self.end_time: Optional[float] = None
    
    def finalize(self, end_time: float):
        self.end_time = end_time
    
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0

class EndOfUtteranceDetector:
    def __init__(self, model_path: str = "smart_turn_v3.onnx"):
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options=so)
        self.audio_buffer = np.array([], dtype=np.float32)
        self.min_samples = 4 * 16000
        self.optimal_samples = 8 * 16000
        self.max_buffer_samples = 8 * 16000
    
    def add_audio_chunk(self, audio_chunk: np.ndarray):
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
        if len(self.audio_buffer) > self.max_buffer_samples:
            self.audio_buffer = self.audio_buffer[-self.max_buffer_samples:]
    
    def has_sufficient_audio(self) -> bool:
        return len(self.audio_buffer) >= self.min_samples
    
    def analyze(self) -> dict:
        if not self.has_sufficient_audio():
            return {'utterance_ended': False, 'confidence': 0.0}
        
        try:
            audio_length = min(len(self.audio_buffer), self.optimal_samples)
            audio = self.audio_buffer[-audio_length:]
            
            inputs = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors="np",
                padding="max_length", max_length=self.optimal_samples,
                truncation=True, do_normalize=True
            )
            
            input_features = inputs.input_features.squeeze(0).astype(np.float32)
            input_features = np.expand_dims(input_features, axis=0)
            
            outputs = self.session.run(None, {"input_features": input_features})
            confidence = float(outputs[0][0].item())
            
            return {
                'utterance_ended': confidence > 0.5,
                'confidence': confidence
            }
        except Exception as e:
            print(f"EoU error: {e}")
            return {'utterance_ended': False, 'confidence': 0.0}
    
    def reset(self):
        self.audio_buffer = np.array([], dtype=np.float32)

class SpeechDetector:
    def __init__(self):
        # VAD
        self.vad_session = ort.InferenceSession('silero_vad.onnx')
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._vad_context = np.zeros((1, 64), dtype=np.float32)
        
        # EOU
        self.eou = None
        try:
            self.eou = EndOfUtteranceDetector("smart_turn_v3.onnx")
            print("EOU model loaded")
        except Exception as e:
            print(f"EOU unavailable: {e}")
        
        # State
        self.state = SpeechState.QUIET
        self.smoothed_prob = 0.0
        self.chunk_count = 0
        
        # Recording
        self.is_recording = False
        self.full_audio: List[np.ndarray] = []
        self.speech_segments: List[SpeechSegment] = []
        self.current_segment: Optional[SpeechSegment] = None
    
    def start_recording(self):
        self.is_recording = True
        self.full_audio = []
        self.speech_segments = []
        self.current_segment = None
        self.chunk_count = 0
        print("Recording started")
    
    def stop_recording(self):
        self.is_recording = False
        
        # Save full audio
        if self.full_audio:
            full_audio_array = np.concatenate(self.full_audio)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"full_recording_{timestamp}.wav"
            self._save_wav(full_audio_array, filename)
            print(f"Full recording saved: {filename}")
        
        # Finalize current segment if any
        if self.current_segment:
            current_time = self.chunk_count * 512 / 16000
            self.current_segment.finalize(current_time)
            self.speech_segments.append(self.current_segment)
            self.current_segment = None
        
        # Print all speech segments
        print("\n=== SPEECH SEGMENTS ===")
        for i, segment in enumerate(self.speech_segments):
            print(f"Segment {i+1}: {segment.start_time:.2f}s - {segment.end_time:.2f}s (Duration: {segment.duration():.2f}s)")
        print(f"Total segments: {len(self.speech_segments)}\n")
    
    def process_chunk(self, audio_chunk: np.ndarray):
        if not self.is_recording:
            return []
        
        self.chunk_count += 1
        current_time = self.chunk_count * 512 / 16000
        
        # Store full audio
        self.full_audio.append(audio_chunk.copy())
        
        # VAD processing
        prob = self._get_vad_probability(audio_chunk)
        self._current_prob = prob  # Store current chunk probability
        self.smoothed_prob = ALPHA * prob + (1.0 - ALPHA) * self.smoothed_prob
        
        # Add to EOU buffer and run analysis only in stopping state
        eou_result = None
        if self.eou and self.state == SpeechState.STOPPING:
            self.eou.add_audio_chunk(audio_chunk)
            if self.eou.has_sufficient_audio():
                eou_result = self.eou.analyze()
                print(f"EOU Analysis - Smoothed Prob: {self.smoothed_prob:.3f}, EOU Confidence: {eou_result['confidence']:.3f}, Utterance Ended: {eou_result['utterance_ended']}")
        elif self.eou and self.state != SpeechState.QUIET:
            # Only add to buffer if not in quiet state, but don't analyze unless stopping
            self.eou.add_audio_chunk(audio_chunk)
        
        # State machine
        events = []
        
        if self.state == SpeechState.QUIET:
            if self.smoothed_prob > THRESHOLD - START_DELTA:    # 0.3 (starting with 0.0)
                self.state = SpeechState.STARTING
                self.current_segment = SpeechSegment(current_time)
                events.append("speech_starting")
                print(f"Starting speech detection at {current_time:.2f}s")
        
        elif self.state == SpeechState.STARTING:
            if self.smoothed_prob > THRESHOLD + START_DELTA:
                self.state = SpeechState.SPEAKING
                events.append("speech_started")
        
        elif self.state == SpeechState.SPEAKING:
            if self.smoothed_prob < THRESHOLD - STOP_DELTA:
                self.state = SpeechState.STOPPING
                events.append("speech_stopping")
        
        elif self.state == SpeechState.STOPPING:
            self.eou.add_audio_chunk(audio_chunk)
            if self.eou and self.eou.has_sufficient_audio():
                eou_result = self.eou.analyze()
                if eou_result['utterance_ended'] and eou_result['confidence'] > 0.9:
                    self.state = SpeechState.QUIET
                    self._finalize_current_segment(current_time, "VAD + Smart Turn Model")
                    events.append("speech_ended")

            if self.state != SpeechState.QUIET and self.smoothed_prob <= FALLBACK_THRESHOLD:
                self.state = SpeechState.QUIET
                self._finalize_current_segment(current_time, "VAD + Fallback Model")
                events.append("speech_ended")
                print(f"Speech ended - EOU confirmed utterance completion")

            if self.smoothed_prob > THRESHOLD - START_DELTA:
                self.state = SpeechState.STARTING
                events.append("speech_started")
        
        return events
    
    def _finalize_current_segment(self, end_time: float, quiet_reason: str):
        if self.current_segment:
            self.current_segment.finalize(end_time)
            self.speech_segments.append(self.current_segment)
            print(f"Segment finalized: {self.current_segment.duration():.2f}s, cause: {quiet_reason}")
            self.current_segment = None
    
    def _determine_quiet_reason(self) -> str:
        # This method is now only used for fallback cases in starting state
        if self.smoothed_prob < FALLBACK_THRESHOLD:
            return "VAD with fallback threshold"
        return "VAD standard threshold"
    
    def _get_vad_probability(self, chunk):
        x = np.concatenate([self._vad_context, chunk.reshape(1, -1)], axis=1)
        out, self._vad_state = self.vad_session.run(
            None, 
            {'input': x, 'state': self._vad_state, 'sr': np.array([16000], dtype=np.int64)}
        )
        self._vad_context = x[:, -64:]
        return float(out[0][0])
    
    def _save_wav(self, audio_array: np.ndarray, filename: str):
        audio_int16 = (audio_array * 32767).astype(np.int16)
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_int16.tobytes())
    
    def get_current_state(self) -> dict:
        return {
            'state': self.state.value,
            'current_prob': float(getattr(self, '_current_prob', 0.0)),
            'smoothed_prob': float(self.smoothed_prob),
            'segments_count': len(self.speech_segments),
            'is_recording': self.is_recording,
            'current_segment_duration': float(
                (self.chunk_count * 512 / 16000) - self.current_segment.start_time 
                if self.current_segment else 0
            )
        }
    
    def reset(self):
        self.state = SpeechState.QUIET
        self.smoothed_prob = 0.0
        self.chunk_count = 0
        self.is_recording = False
        self.full_audio = []
        self.speech_segments = []
        self.current_segment = None
        
        if self.eou:
            self.eou.reset()
        
        print("System reset")

# FastAPI app
app = FastAPI()

@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    detector = SpeechDetector()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "audio":
                audio_bytes = base64.b64decode(message["data"])
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                
                if len(audio_array) == 512:
                    events = detector.process_chunk(audio_array)
                    
                    if events:
                        response = {
                            "events": events,
                            "state": detector.get_current_state()
                        }
                        await websocket.send_text(json.dumps(response))
            
            elif message["type"] == "start_recording":
                detector.start_recording()
                await websocket.send_text(json.dumps({"recording_started": True}))
            
            elif message["type"] == "stop_recording":
                detector.stop_recording()
                await websocket.send_text(json.dumps({"recording_stopped": True}))
            
            elif message["type"] == "reset":
                detector.reset()
                await websocket.send_text(json.dumps({"reset": "ok"}))
                
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)