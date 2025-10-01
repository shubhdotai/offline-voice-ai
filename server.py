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
import asyncio

# Configuration
ALPHA = 0.2
START_THRESHOLD = 0.3
SPEAKING_THRESHOLD = 0.5
STOP_THRESHOLD = 0.3
QUIET_THRESHOLD = 0.05

# Feature flags
ENABLE_RECORDING = False
ENABLE_TRANSCRIPTION = True

# Safety margins
SAFETY_CHUNKS_BEFORE = 4    # 128ms
SAFETY_CHUNKS_AFTER = 2

# Queue configuration
MAX_TRANSCRIPTION_QUEUE_SIZE = 2  # Only keep 2 pending transcriptions max

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
            self.active_segment.append(chunk.copy())
            # Also maintain pre_buffer rolling window
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
    
    def get_segment_copy(self) -> Optional[np.ndarray]:
        """Get copy of current segment without resetting."""
        if not self.active_segment:
            return None
        return np.concatenate(self.active_segment)
    
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

class TranscriptionRequest:
    """Container for transcription request data."""
    def __init__(self, segment_id: int, audio_data: np.ndarray, initial_prompt: str = ""):
        self.segment_id = segment_id
        self.audio_data = audio_data
        self.initial_prompt = initial_prompt
        self.timestamp = datetime.now()

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
        
        # Transcription queue system
        self.transcription_queue = asyncio.Queue(maxsize=MAX_TRANSCRIPTION_QUEUE_SIZE)
        self.transcription_started = False
        self.captured_segment = None
        self.dropped_segments = 0
    
    def _warmup_transcriber(self):
        """Warm up Whisper model."""
        print("Warming up transcription model...")
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.001
        try:
            self.transcriber.transcribe(dummy_audio)
            print("Transcription model ready")
        except Exception as e:
            print(f"Warmup warning: {e}")
    
    def start_listening(self):
        self.is_listening = True
        self.full_audio = []
        self.segment_count = 0
        self.chunk_count = 0
        self.dropped_segments = 0
        print("Listening started")
    
    def stop_listening(self):
        self.is_listening = False
        
        if ENABLE_RECORDING and self.full_audio:
            full_audio_array = np.concatenate(self.full_audio)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            self._save_wav(full_audio_array, filename)
            print(f"Recording saved: {filename}")
        
        if self.dropped_segments > 0:
            print(f"Warning: Dropped {self.dropped_segments} segments due to queue overflow")
        
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
                # False start
                self.state = SpeechState.QUIET
                self.segment_buffer.reset()
        
        elif self.state == SpeechState.SPEAKING:
            if self.smoothed_prob < STOP_THRESHOLD:
                self.state = SpeechState.STOPPING
                events.append("speech_stopping")
                # Mark that we should start transcription (only once)
                if not self.transcription_started:
                    self.transcription_started = True
                    # Capture the segment NOW and CLEAR it immediately
                    self.captured_segment = self.segment_buffer.get_segment()
                    events.append("start_transcription")
        
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
                self.state = SpeechState.QUIET
                events.append("speech_ended")
                
                # Reset transcription flag for next segment
                self.transcription_started = False
                self.captured_segment = None
                
                if self.eou:
                    self.eou.reset()
            
            # Resume speaking if VAD increases
            elif self.smoothed_prob > SPEAKING_THRESHOLD:
                self.state = SpeechState.SPEAKING
                events.append("speech_resumed")
                self.segment_buffer.post_buffer_count = 0
                # CRITICAL: If resuming, the old segment was already captured and queued
                # Start fresh for the next segment
                self.transcription_started = False
                self.captured_segment = None
                # Note: segment_buffer.active_segment already cleared by get_segment() call
        
        return events
    
    async def queue_transcription(self) -> bool:
        """Queue a transcription request. Returns False if queue is full."""
        if not ENABLE_TRANSCRIPTION or not self.transcriber:
            return False
        
        segment_array = self.captured_segment
        
        if segment_array is None or len(segment_array) < 16000 * 0.3:
            print(f"Segment too short, skipping")
            return False
        
        self.segment_count += 1
        
        # Create transcription request
        request = TranscriptionRequest(
            segment_id=self.segment_count,
            audio_data=segment_array,
            initial_prompt=""
        )
        
        # Try to add to queue (non-blocking)
        try:
            self.transcription_queue.put_nowait(request)
            print(f"Segment {request.segment_id}: Queued for transcription (length: {len(segment_array)/16000:.2f}s, queue size: {self.transcription_queue.qsize()})")
            return True
        except asyncio.QueueFull:
            # Queue is full - drop this segment
            self.dropped_segments += 1
            print(f"⚠️  Queue full! Dropped segment {request.segment_id} (total dropped: {self.dropped_segments})")
            return False
    
    async def process_transcription_queue(self, websocket: WebSocket):
        """Consumer coroutine that processes transcription queue."""
        print("Transcription worker started")
        
        while True:
            try:
                # Wait for next transcription request
                request = await self.transcription_queue.get()
                
                print(f"Segment {request.segment_id}: Transcription started (queue size: {self.transcription_queue.qsize()})")
                
                # Run transcription in executor to avoid blocking
                loop = asyncio.get_event_loop()
                transcript = await loop.run_in_executor(
                    None,
                    self.transcriber.transcribe,
                    request.audio_data
                )
                
                # Send result if valid
                if transcript and transcript.strip():
                    print(f"Segment {request.segment_id}: {transcript}")
                    
                    # Send to client
                    response = {
                        "events": ["transcript"],
                        "transcript": transcript,
                        "segment_id": request.segment_id,
                        "state": self.get_current_state()
                    }
                    await websocket.send_text(json.dumps(response))
                else:
                    print(f"Segment {request.segment_id}: Empty transcript, skipping")
                
                # Mark task as done
                self.transcription_queue.task_done()
                
            except Exception as e:
                print(f"Transcription worker error: {e}")
                # Continue processing even if one transcription fails
                continue
    
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
        self.segment_buffer.reset()
        self.dropped_segments = 0
        
        # Clear queue
        while not self.transcription_queue.empty():
            try:
                self.transcription_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
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
    
    # Start transcription worker in background
    transcription_worker = asyncio.create_task(
        detector.process_transcription_queue(websocket)
    )
    
    try:
        while True:
            message = await websocket.receive()
            
            # Binary audio data
            if "bytes" in message:
                audio_array = np.frombuffer(message["bytes"], dtype=np.float32)
                
                if len(audio_array) == 512:
                    events = detector.process_chunk(audio_array)
                    
                    # Queue transcription if needed
                    if "start_transcription" in events:
                        await detector.queue_transcription()
                    
                    # Send state updates
                    if events:
                        response = {
                            "events": [e for e in events if e != "start_transcription"],
                            "state": detector.get_current_state()
                        }
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
        transcription_worker.cancel()
    except Exception as e:
        print(f"WebSocket error: {e}")
        transcription_worker.cancel()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)