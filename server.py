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
from llm_handler import LLMHandler
from tts_handler import TTSHandler
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
SAFETY_CHUNKS_BEFORE = 4
SAFETY_CHUNKS_AFTER = 2

# Queue configuration
MAX_TRANSCRIPTION_QUEUE_SIZE = 2

class SpeechState(Enum):
    QUIET = "quiet"
    STARTING = "starting"
    SPEAKING = "speaking"
    STOPPING = "stopping"

class EndOfUtteranceDetector:
    def __init__(self, model_path: str = "models/smart_turn_v3.onnx"):
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
    def __init__(self, safety_before: int = SAFETY_CHUNKS_BEFORE, 
                 safety_after: int = SAFETY_CHUNKS_AFTER):
        self.safety_before = safety_before
        self.safety_after = safety_after
        self.pre_buffer = []
        self.active_segment = []
        self.post_buffer_count = 0
        self.is_capturing = False
        
    def add_chunk(self, chunk: np.ndarray, state: SpeechState):
        if state == SpeechState.QUIET:
            self.pre_buffer.append(chunk.copy())
            if len(self.pre_buffer) > self.safety_before:
                self.pre_buffer.pop(0)
            self.post_buffer_count = 0
                
        elif state == SpeechState.STARTING:
            if not self.is_capturing:
                self.is_capturing = True
                self.active_segment = [c.copy() for c in self.pre_buffer]
            self.active_segment.append(chunk.copy())
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
        return self.post_buffer_count >= self.safety_after
    
    def get_segment_copy(self) -> Optional[np.ndarray]:
        if not self.active_segment:
            return None
        return np.concatenate(self.active_segment)
    
    def get_segment(self) -> Optional[np.ndarray]:
        if not self.active_segment:
            return None
        
        segment = np.concatenate(self.active_segment)
        self.reset()
        return segment
    
    def reset(self):
        self.active_segment = []
        self.post_buffer_count = 0
        self.is_capturing = False

class TranscriptionRequest:
    def __init__(self, segment_id: int, audio_data: np.ndarray, initial_prompt: str = ""):
        self.segment_id = segment_id
        self.audio_data = audio_data
        self.initial_prompt = initial_prompt
        self.timestamp = datetime.now()

class SpeechDetector:
    def __init__(self):
        # VAD
        self.vad_session = ort.InferenceSession('models/silero_vad.onnx')
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._vad_context = np.zeros((1, 64), dtype=np.float32)
        
        # EOU
        self.eou = None
        try:
            self.eou = EndOfUtteranceDetector("models/smart_turn_v3.onnx")
            print("EOU model loaded")
        except Exception as e:
            print(f"EOU unavailable: {e}")
        
        # Transcriber
        self.transcriber = None
        if ENABLE_TRANSCRIPTION:
            self.transcriber = RealtimeTranscriber()
            self._warmup_transcriber()
        
        # LLM and TTS
        self.llm_handler = LLMHandler()
        self.tts_handler = TTSHandler()
        
        # State
        self.state = SpeechState.QUIET
        self.smoothed_prob = 0.0
        self.current_prob = 0.0
        self.chunk_count = 0
        
        # Buffers
        self.segment_buffer = SpeechSegmentBuffer()
        self.is_listening = False
        self.is_playing_response = False
        self.full_audio: List[np.ndarray] = []
        self.segment_count = 0
        
        # Transcription queue
        self.transcription_queue = asyncio.Queue(maxsize=MAX_TRANSCRIPTION_QUEUE_SIZE)
        self.transcription_started = False
        self.captured_segment = None
        self.dropped_segments = 0
        
        # Conversation history
        self.conversation_history = []
    
    def _warmup_transcriber(self):
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
        self.chunk_count += 1
        
        # Get VAD probability
        prob = self._get_vad_probability(audio_chunk)
        self.current_prob = prob
        self.smoothed_prob = ALPHA * prob + (1.0 - ALPHA) * self.smoothed_prob
        
        # If not listening or playing response, just update VAD
        if not self.is_listening or self.is_playing_response:
            return []
        
        # Store for recording
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
                self.state = SpeechState.QUIET
                self.segment_buffer.reset()
        
        elif self.state == SpeechState.SPEAKING:
            if self.smoothed_prob < STOP_THRESHOLD:
                self.state = SpeechState.STOPPING
                events.append("speech_stopping")
                if not self.transcription_started:
                    self.transcription_started = True
                    self.captured_segment = self.segment_buffer.get_segment()
                    events.append("start_transcription")
        
        elif self.state == SpeechState.STOPPING:
            should_finalize = False
            
            if self.eou and self.eou.has_sufficient_audio():
                eou_result = self.eou.analyze()
                if eou_result['utterance_ended'] and eou_result['confidence'] > 0.9:
                    should_finalize = True
            
            if not should_finalize and self.segment_buffer.has_safety_margin():
                if self.smoothed_prob < QUIET_THRESHOLD:
                    should_finalize = True
            
            if should_finalize:
                self.state = SpeechState.QUIET
                events.append("speech_ended")
                events.append("process_response")  # Trigger LLM + TTS
                
                self.transcription_started = False
                self.captured_segment = None
                
                if self.eou:
                    self.eou.reset()
            
            elif self.smoothed_prob > SPEAKING_THRESHOLD:
                self.state = SpeechState.SPEAKING
                events.append("speech_resumed")
                self.segment_buffer.post_buffer_count = 0
                self.transcription_started = False
                self.captured_segment = None
        
        return events
    
    async def queue_transcription(self) -> bool:
        if not ENABLE_TRANSCRIPTION or not self.transcriber:
            return False
        
        segment_array = self.captured_segment
        
        if segment_array is None or len(segment_array) < 16000 * 0.3:
            print(f"Segment too short, skipping")
            return False
        
        self.segment_count += 1
        
        request = TranscriptionRequest(
            segment_id=self.segment_count,
            audio_data=segment_array,
            initial_prompt=""
        )
        
        try:
            self.transcription_queue.put_nowait(request)
            print(f"Segment {request.segment_id}: Queued for transcription (length: {len(segment_array)/16000:.2f}s)")
            return True
        except asyncio.QueueFull:
            self.dropped_segments += 1
            print(f"⚠️  Queue full! Dropped segment {request.segment_id}")
            return False
    
    async def process_transcription_queue(self, websocket: WebSocket):
        print("Transcription worker started")
        
        while True:
            try:
                request = await self.transcription_queue.get()
                
                print(f"Segment {request.segment_id}: Transcription started")
                
                loop = asyncio.get_event_loop()
                transcript = await loop.run_in_executor(
                    None,
                    self.transcriber.transcribe,
                    request.audio_data
                )
                
                if transcript and transcript.strip():
                    print(f"User said: {transcript}")
                    
                    # Add to conversation history
                    self.conversation_history.append({
                        "role": "user",
                        "content": transcript
                    })
                    
                    # Send transcript to client
                    response = {
                        "events": ["transcript"],
                        "transcript": transcript,
                        "segment_id": request.segment_id,
                        "state": self.get_current_state()
                    }
                    await websocket.send_text(json.dumps(response))
                else:
                    print(f"Segment {request.segment_id}: Empty transcript, skipping")
                
                self.transcription_queue.task_done()
                
            except Exception as e:
                print(f"Transcription worker error: {e}")
                continue
    
    async def process_llm_and_tts(self, websocket: WebSocket):
        """Process the conversation through LLM and TTS when quiet event occurs."""
        try:
            # Block new speech input
            self.is_playing_response = True
            
            # Check if we have any user input to respond to
            user_messages = [msg for msg in self.conversation_history if msg['role'] == 'user']
            if not user_messages:
                print("No user input to respond to")
                self.is_playing_response = False
                return
            
            # Notify client that processing has started
            await websocket.send_text(json.dumps({
                "events": ["processing_started"],
                "state": self.get_current_state()
            }))
            
            # Get LLM response
            print("Generating LLM response...")
            loop = asyncio.get_event_loop()
            
            try:
                llm_response = await loop.run_in_executor(
                    None,
                    self.llm_handler.generate_response,
                    self.conversation_history
                )
            except Exception as llm_error:
                print(f"LLM generation failed: {llm_error}")
                llm_response = "I apologize, I'm having trouble processing that right now."
            
            if not llm_response or not llm_response.strip():
                print("Empty LLM response, skipping")
                self.is_playing_response = False
                return
            
            print(f"Assistant: {llm_response}")
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": llm_response
            })
            
            # Generate TTS audio
            print("Generating TTS audio...")
            try:
                audio_bytes = await loop.run_in_executor(
                    None,
                    self.tts_handler.generate_speech,
                    llm_response
                )
            except Exception as tts_error:
                print(f"TTS generation failed: {tts_error}")
                audio_bytes = None
            
            if not audio_bytes:
                print("Failed to generate TTS audio")
                self.is_playing_response = False
                return
            
            # Send audio to client
            print(f"Sending audio response ({len(audio_bytes)} bytes)")
            await websocket.send_bytes(audio_bytes)
            
            # Send completion event
            await websocket.send_text(json.dumps({
                "events": ["response_complete"],
                "llm_response": llm_response,
                "state": self.get_current_state()
            }))
            
        except Exception as e:
            print(f"Error in LLM/TTS processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Re-enable speech input
            self.is_playing_response = False
    
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
            'smoothed_prob': float(self.smoothed_prob),
            'segments_count': self.segment_count,
            'is_listening': self.is_listening,
            'is_playing_response': self.is_playing_response,
        }
    
    def reset(self):
        self.state = SpeechState.QUIET
        self.smoothed_prob = 0.0
        self.current_prob = 0.0
        self.chunk_count = 0
        self.is_listening = False
        self.is_playing_response = False
        self.full_audio = []
        self.segment_count = 0
        self.segment_buffer.reset()
        self.dropped_segments = 0
        self.conversation_history = []
        
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
    
    transcription_worker = asyncio.create_task(
        detector.process_transcription_queue(websocket)
    )
    
    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message:
                audio_array = np.frombuffer(message["bytes"], dtype=np.float32)
                
                if len(audio_array) == 512:
                    events = detector.process_chunk(audio_array)
                    
                    if "start_transcription" in events:
                        await detector.queue_transcription()
                    
                    if "process_response" in events:
                        await detector.process_llm_and_tts(websocket)
                    
                    if events:
                        filtered_events = [e for e in events if e not in ["start_transcription", "process_response"]]
                        if filtered_events:
                            response = {
                                "events": filtered_events,
                                "state": detector.get_current_state()
                            }
                            await websocket.send_text(json.dumps(response))
            
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
                
                elif data["type"] == "audio_finished":
                    # Client finished playing audio
                    print("Client finished playing audio")
                
    except WebSocketDisconnect:
        print("Client disconnected")
        transcription_worker.cancel()
    except Exception as e:
        print(f"WebSocket error: {e}")
        transcription_worker.cancel()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)