import json
import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import wave
from datetime import datetime
from enum import Enum
from typing import Optional, List
from transcriber import RealtimeTranscriber

# Configuration
ALPHA = 0.1
START_THRESHOLD = 0.3
SPEAKING_THRESHOLD = 0.5
STOP_THRESHOLD = 0.2
QUIET_THRESHOLD = 0.1

# Feature flags
ENABLE_RECORDING = False
ENABLE_TRANSCRIPTION = True

# Safety margins
SAFETY_CHUNKS_BEFORE = 2
SAFETY_CHUNKS_AFTER = 2

class SpeechState(Enum):
    QUIET = "quiet"
    STARTING = "starting"
    SPEAKING = "speaking"
    STOPPING = "stopping"

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

class SpeechSegmentBuffer:
    """Manages audio buffering with safety margins."""
    
    def __init__(self, safety_before: int = SAFETY_CHUNKS_BEFORE, 
                 safety_after: int = SAFETY_CHUNKS_AFTER):
        self.safety_before = safety_before
        self.safety_after = safety_after
        self.pre_buffer = []
        self.active_segment = []
        self.post_buffer_count = 0
        self.is_capturing = False
        
    def add_chunk(self, chunk: np.ndarray, state: SpeechState):
        """Add audio chunk based on current state."""
        if state == SpeechState.QUIET:
            # Always maintain rolling pre-buffer for next speech segment
            self.pre_buffer.append(chunk.copy())
            if len(self.pre_buffer) > self.safety_before:
                self.pre_buffer.pop(0)
            self.post_buffer_count = 0
                
        elif state == SpeechState.STARTING:
            if not self.is_capturing:
                # Start capturing with ALL pre-buffer chunks
                self.is_capturing = True
                self.active_segment = [c.copy() for c in self.pre_buffer]
                # Keep pre_buffer intact for continuous rolling window
            self.active_segment.append(chunk.copy())
            # Also add to pre_buffer to maintain rolling window
            self.pre_buffer.append(chunk.copy())
            if len(self.pre_buffer) > self.safety_before:
                self.pre_buffer.pop(0)
            self.post_buffer_count = 0
            
        elif state == SpeechState.SPEAKING:
            self.active_segment.append(chunk.copy())
            self.post_buffer_count = 0
            
        elif state == SpeechState.STOPPING:
            self.active_segment.append(chunk.copy())
            self.post_buffer_count += 1
    
    def has_safety_margin(self) -> bool:
        """Check if we've collected enough post-stop chunks."""
        return self.post_buffer_count >= self.safety_after
    
    def get_segment(self) -> Optional[np.ndarray]:
        """Get complete segment and reset."""
        if not self.active_segment:
            return None
        
        segment = np.concatenate(self.active_segment)
        self.reset()
        return segment
    
    def reset(self):
        """Reset segment capture but preserve pre_buffer for next segment."""
        self.active_segment = []
        self.post_buffer_count = 0
        self.is_capturing = False
        # Note: pre_buffer is NOT reset to maintain continuity

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
        
        # Transcriber with warm-up
        self.transcriber = None
        self.last_transcript = ""
        if ENABLE_TRANSCRIPTION:
            self.transcriber = RealtimeTranscriber()
            self._warmup_transcriber()
        
        # State
        self.state = SpeechState.QUIET
        self.smoothed_prob = 0.0
        self.current_prob = 0.0
        self.chunk_count = 0
        
        # Buffers
        self.segment_buffer = SpeechSegmentBuffer()
        self.is_listening = False
        self.full_audio: List[np.ndarray] = []
        self.segment_count = 0
    
    def _warmup_transcriber(self):
        """Warm up Whisper model."""
        print("Warming up transcription model...")
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.001
        try:
            self.transcriber.transcribe(dummy_audio, initial_prompt="")
            print("Transcription model ready")
        except Exception as e:
            print(f"Warmup warning: {e}")
    
    def start_listening(self):
        self.is_listening = True
        self.full_audio = []
        self.segment_count = 0
        self.chunk_count = 0
        self.last_transcript = ""
        print("Listening started")
    
    def stop_listening(self):
        self.is_listening = False
        
        if ENABLE_RECORDING and self.full_audio:
            full_audio_array = np.concatenate(self.full_audio)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            self._save_wav(full_audio_array, filename)
            print(f"Recording saved: {filename}")
        
        print(f"Total segments: {self.segment_count}\n")
    
    def process_chunk(self, audio_chunk: np.ndarray):
        """Process audio chunk - called continuously regardless of listening state."""
        self.chunk_count += 1
        
        # Get VAD probability
        prob = self._get_vad_probability(audio_chunk)
        self.current_prob = prob
        self.smoothed_prob = ALPHA * prob + (1.0 - ALPHA) * self.smoothed_prob
        
        # If not listening, just update VAD and return
        if not self.is_listening:
            return []
        
        # Store for recording if enabled
        if ENABLE_RECORDING:
            self.full_audio.append(audio_chunk.copy())
        
        # Add to segment buffer
        self.segment_buffer.add_chunk(audio_chunk, self.state)
        
        # Add to EOU buffer when speaking/stopping
        if self.eou and self.state in [SpeechState.SPEAKING, SpeechState.STOPPING]:
            self.eou.add_audio_chunk(audio_chunk)
        
        # State machine
        return self._update_state()
    
    def _update_state(self):
        """Update state machine and return events."""
        events = []
        
        if self.state == SpeechState.QUIET:
            if self.smoothed_prob >= START_THRESHOLD:
                self.state = SpeechState.STARTING
                events.append("speech_starting")
        
        elif self.state == SpeechState.STARTING:
            if self.smoothed_prob >= SPEAKING_THRESHOLD:
                self.state = SpeechState.SPEAKING
                events.append("speech_started")
            elif self.smoothed_prob < QUIET_THRESHOLD:
                # False start - reset transcript context and buffer
                self.state = SpeechState.QUIET
                self.segment_buffer.reset()
                self.last_transcript = ""
        
        elif self.state == SpeechState.SPEAKING:
            if self.smoothed_prob < STOP_THRESHOLD:
                self.state = SpeechState.STOPPING
                events.append("speech_stopping")
        
        elif self.state == SpeechState.STOPPING:
            # Check if should finalize
            should_finalize = False
            
            # Priority 1: EOU detection
            if self.eou and self.eou.has_sufficient_audio():
                eou_result = self.eou.analyze()
                if eou_result['utterance_ended'] and eou_result['confidence'] > 0.9:
                    should_finalize = True
            
            # Priority 2: After safety margin, check if truly quiet
            if not should_finalize and self.segment_buffer.has_safety_margin():
                if self.smoothed_prob < QUIET_THRESHOLD:
                    should_finalize = True
            
            if should_finalize:
                # Determine if this is end of utterance or just a pause
                going_to_quiet = self.smoothed_prob < QUIET_THRESHOLD
                
                # Process segment with appropriate context
                transcript = self._process_segment(
                    use_context=not going_to_quiet  # Use context only if continuing
                )
                
                self.state = SpeechState.QUIET
                events.append("speech_ended")
                
                # Reset context if going to quiet (full stop)
                if going_to_quiet:
                    self.last_transcript = ""
                
                if transcript:
                    events.append(("transcript", transcript))
                
                if self.eou:
                    self.eou.reset()
            
            # Resume speaking if VAD increases
            elif self.smoothed_prob > SPEAKING_THRESHOLD:
                self.state = SpeechState.SPEAKING
                events.append("speech_resumed")
                self.segment_buffer.post_buffer_count = 0
        
        return events
    
    def _process_segment(self, use_context: bool = True) -> Optional[str]:
        """Process completed speech segment.
        
        Args:
            use_context: If True, use last_transcript as initial_prompt for continuity.
                        If False, transcribe without context (clean slate).
        """
        self.segment_count += 1
        
        if not ENABLE_TRANSCRIPTION or not self.transcriber:
            self.segment_buffer.reset()
            return None
        
        segment_array = self.segment_buffer.get_segment()
        
        if segment_array is None or len(segment_array) < 16000 * 0.3:
            print(f"Segment {self.segment_count}: Too short")
            return None
        
        # Transcribe with or without context
        initial_prompt = self.last_transcript if use_context else ""
        transcript = self.transcriber.transcribe(segment_array, initial_prompt=initial_prompt)
        
        if transcript:
            context_info = "with context" if use_context and initial_prompt else "no context"
            print(f"Segment {self.segment_count} ({context_info}): {transcript}")
            
            # Update context for next transcription (will be used if use_context=True)
            self.last_transcript = transcript[-200:] if len(transcript) > 200 else transcript
        
        return transcript
    
    def _get_vad_probability(self, chunk):
        """Get VAD probability for chunk."""
        x = np.concatenate([self._vad_context, chunk.reshape(1, -1)], axis=1)
        out, self._vad_state = self.vad_session.run(
            None, 
            {'input': x, 'state': self._vad_state, 'sr': np.array([16000], dtype=np.int64)}
        )
        self._vad_context = x[:, -64:]
        return float(out[0][0])
    
    def _save_wav(self, audio_array: np.ndarray, filename: str):
        """Save audio to WAV file."""
        audio_int16 = (audio_array * 32767).astype(np.int16)
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_int16.tobytes())
    
    def get_current_state(self) -> dict:
        """Get current state for UI."""
        return {
            'state': self.state.value,
            'current_prob': float(self.current_prob),
            'smoothed_prob': float(self.smoothed_prob),
            'segments_count': self.segment_count,
            'is_listening': self.is_listening,
        }
    
    def reset(self):
        """Reset detector state."""
        self.state = SpeechState.QUIET
        self.smoothed_prob = 0.0
        self.current_prob = 0.0
        self.chunk_count = 0
        self.is_listening = False
        self.full_audio = []
        self.segment_count = 0
        self.last_transcript = ""
        self.segment_buffer.reset()
        
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
            message = await websocket.receive()
            
            # Binary audio data
            if "bytes" in message:
                audio_array = np.frombuffer(message["bytes"], dtype=np.float32)
                
                if len(audio_array) == 512:
                    events = detector.process_chunk(audio_array)
                    
                    # Always send state updates, even if no events
                    response = {
                        "events": [e if isinstance(e, str) else e[0] for e in events] if events else [],
                        "state": detector.get_current_state()
                    }
                    
                    # Add transcript if present
                    for event in events:
                        if isinstance(event, tuple) and event[0] == "transcript":
                            response["transcript"] = event[1]
                    
                    await websocket.send_text(json.dumps(response))
            
            # Text commands
            elif "text" in message:
                data = json.loads(message["text"])
                
                if data["type"] == "start_listening":
                    detector.start_listening()
                    await websocket.send_text(json.dumps({"listening_started": True}))
                
                elif data["type"] == "stop_listening":
                    detector.stop_listening()
                    await websocket.send_text(json.dumps({"listening_stopped": True}))
                
                elif data["type"] == "reset":
                    detector.reset()
                    await websocket.send_text(json.dumps({"reset": "ok"}))
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)