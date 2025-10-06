import json
import base64
import asyncio
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from transformers import WhisperFeatureExtractor

from llm_handler import LLMHandler
from transcriber import RealtimeTranscriber
from tts_handler import TTSHandler
import wave

# Configuration
ALPHA = 0.2
START_THRESHOLD = 0.3
SPEAKING_THRESHOLD = 0.5
STOP_THRESHOLD = 0.3
QUIET_THRESHOLD = 0.05
SAFETY_CHUNKS_BEFORE = 4
SAFETY_CHUNKS_AFTER = 2
MAX_TRANSCRIPTION_QUEUE_SIZE = 2

# Feature flags
ENABLE_RECORDING = False
ENABLE_TRANSCRIPTION = True


def _encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _decode_bytes(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def _decode_float32_chunk(data: str) -> Optional[np.ndarray]:
    try:
        audio_bytes = _decode_bytes(data)
        if len(audio_bytes) % 4 != 0:
            return None
        return np.frombuffer(audio_bytes, dtype=np.float32)
    except Exception:
        return None


def _split_sentences(text: str) -> List[str]:
    sentences, buffer = [], []
    for ch in text:
        buffer.append(ch)
        if ch in ".!?":
            chunk = "".join(buffer).strip()
            if chunk:
                sentences.append(chunk)
            buffer = []
    remainder = "".join(buffer).strip()
    if remainder:
        sentences.append(remainder)
    return sentences


class SpeechState(Enum):
    QUIET = "quiet"
    STARTING = "starting"
    SPEAKING = "speaking"
    STOPPING = "stopping"


class PipelineEvent(str, Enum):
    SPEECH_STARTING = "speech_starting"
    SPEECH_STARTED = "speech_started"
    SPEECH_STOPPING = "speech_stopping"
    SPEECH_ENDED = "speech_ended"
    SPEECH_RESUMED = "speech_resumed"
    START_TRANSCRIPTION = "start_transcription"
    PROCESS_RESPONSE = "process_response"


class PipelineResources:
    def __init__(self):
        print("Initializing pipeline resources...")
        self.transcriber = self._load_transcriber()
        self.llm_handler = LLMHandler()
        self.tts_handler = TTSHandler()
        print("Pipeline resources ready")

    def _load_transcriber(self) -> Optional[RealtimeTranscriber]:
        if not ENABLE_TRANSCRIPTION:
            return None
        transcriber = RealtimeTranscriber()
        dummy_audio = np.random.randn(16000).astype(np.float32) * 0.001
        try:
            transcriber.transcribe(dummy_audio)
            print("Transcription model warmed up")
        except Exception as exc:
            print(f"Transcriber warmup warning: {exc}")
        return transcriber


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
            input_features = np.expand_dims(inputs.input_features.squeeze(0).astype(np.float32), axis=0)
            outputs = self.session.run(None, {"input_features": input_features})
            confidence = float(outputs[0][0].item())
            return {'utterance_ended': confidence > 0.5, 'confidence': confidence}
        except Exception as e:
            print(f"EoU error: {e}")
            return {'utterance_ended': False, 'confidence': 0.0}
    
    def reset(self):
        self.audio_buffer = np.array([], dtype=np.float32)


class SpeechSegmentBuffer:
    def __init__(self, safety_before: int = SAFETY_CHUNKS_BEFORE, safety_after: int = SAFETY_CHUNKS_AFTER):
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
    def __init__(self, resources: PipelineResources):
        self.resources = resources
        self.vad_session = ort.InferenceSession('models/silero_vad.onnx')
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)
        self._vad_context = np.zeros((1, 64), dtype=np.float32)
        
        self.eou = None
        try:
            self.eou = EndOfUtteranceDetector("models/smart_turn_v3.onnx")
            print("EOU model loaded")
        except Exception as e:
            print(f"EOU unavailable: {e}")
        
        self.transcriber = resources.transcriber
        self.llm_handler = resources.llm_handler
        self.tts_handler = resources.tts_handler
        
        self.state = SpeechState.QUIET
        self.smoothed_prob = 0.0
        self.current_prob = 0.0
        self.chunk_count = 0
        
        self.segment_buffer = SpeechSegmentBuffer()
        self.is_listening = False
        self.is_playing_response = False
        self.full_audio: List[np.ndarray] = []
        self.segment_count = 0
        
        self.transcription_queue = asyncio.Queue(maxsize=MAX_TRANSCRIPTION_QUEUE_SIZE)
        self.transcription_started = False
        self.captured_segment = None
        self.dropped_segments = 0
        
        self.conversation_history = []

    def _set_state(self, new_state: SpeechState):
        if new_state != self.state:
            print(f"[state] {self.state.value} -> {new_state.value} | vad={self.smoothed_prob:.3f}")
            self.state = new_state
    
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
            self._save_wav(full_audio_array, f"recording_{timestamp}.wav")
            print(f"Recording saved: recording_{timestamp}.wav")
        if self.dropped_segments > 0:
            print(f"Warning: Dropped {self.dropped_segments} segments due to queue overflow")
        print(f"Total segments: {self.segment_count}\n")
    
    def process_chunk(self, audio_chunk: np.ndarray) -> List[PipelineEvent]:
        self.chunk_count += 1
        prob = self._get_vad_probability(audio_chunk)
        self.current_prob = prob
        self.smoothed_prob = ALPHA * prob + (1.0 - ALPHA) * self.smoothed_prob
        
        if not self.is_listening:
            return []
        
        if ENABLE_RECORDING:
            self.full_audio.append(audio_chunk.copy())
        
        self.segment_buffer.add_chunk(audio_chunk, self.state)
        
        if self.eou and self.state in [SpeechState.SPEAKING, SpeechState.STOPPING]:
            self.eou.add_audio_chunk(audio_chunk)
        
        return self._update_state()
    
    def _update_state(self) -> List[PipelineEvent]:
        events: List[PipelineEvent] = []
        
        if self.state == SpeechState.QUIET:
            if self.smoothed_prob >= START_THRESHOLD:
                self._set_state(SpeechState.STARTING)
                events.append(PipelineEvent.SPEECH_STARTING)
                # User is interrupting if bot is currently responding
                if self.is_playing_response:
                    print("[interrupt] User interrupting bot response")
                    # Interruption will be handled by orchestrator
        elif self.state == SpeechState.STARTING:
            if self.smoothed_prob >= SPEAKING_THRESHOLD:
                self._set_state(SpeechState.SPEAKING)
                events.append(PipelineEvent.SPEECH_STARTED)
            elif self.smoothed_prob < QUIET_THRESHOLD:
                self._set_state(SpeechState.QUIET)
                self.segment_buffer.reset()
        elif self.state == SpeechState.SPEAKING:
            if self.smoothed_prob < STOP_THRESHOLD:
                self._set_state(SpeechState.STOPPING)
                events.append(PipelineEvent.SPEECH_STOPPING)
                if not self.transcription_started:
                    self.transcription_started = True
                    self.captured_segment = self.segment_buffer.get_segment()
                    if self.captured_segment is not None:
                        print(f"[audio] Segment ready ({len(self.captured_segment)/16000:.2f}s) for transcription")
                    events.append(PipelineEvent.START_TRANSCRIPTION)
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
                self._set_state(SpeechState.QUIET)
                events.extend([PipelineEvent.SPEECH_ENDED, PipelineEvent.PROCESS_RESPONSE])
                self.transcription_started = False
                self.captured_segment = None
                if self.eou:
                    self.eou.reset()
            elif self.smoothed_prob > SPEAKING_THRESHOLD:
                self._set_state(SpeechState.SPEAKING)
                events.append(PipelineEvent.SPEECH_RESUMED)
                self.segment_buffer.post_buffer_count = 0
                self.transcription_started = False
                self.captured_segment = None
        return events
    
    async def queue_transcription(self) -> bool:
        if not ENABLE_TRANSCRIPTION or not self.transcriber or self.captured_segment is None:
            return False
        if len(self.captured_segment) < 16000 * 0.3:
            print("Segment too short, skipping")
            return False
        self.segment_count += 1
        request = TranscriptionRequest(self.segment_count, self.captured_segment)
        try:
            self.transcription_queue.put_nowait(request)
            print(f"Segment {request.segment_id}: Queued for transcription (length: {len(self.captured_segment)/16000:.2f}s)")
            return True
        except asyncio.QueueFull:
            self.dropped_segments += 1
            print(f"⚠️  Queue full! Dropped segment {request.segment_id}")
            return False
    
    async def process_transcription_queue(self, websocket: WebSocket):
        print("Transcription worker started")
        if not self.transcriber:
            print("Transcription disabled; worker exiting")
            return
        while True:
            request = await self.transcription_queue.get()
            stt_latency = None
            try:
                print(f"Segment {request.segment_id}: Transcription started")
                await self._send_metrics(websocket, stt={"status": "running", "segment_id": request.segment_id})
                stt_start = time.monotonic()
                transcript = await asyncio.get_event_loop().run_in_executor(
                    None, self.transcriber.transcribe, request.audio_data
                )
                stt_latency = time.monotonic() - stt_start
                if transcript and transcript.strip():
                    print(f"User said: {transcript}")
                    self.conversation_history.append({"role": "user", "content": transcript})
                    await websocket.send_text(json.dumps({
                        "event": "text", "role": "user", "text": transcript, "segment_id": request.segment_id
                    }))
                else:
                    print(f"Segment {request.segment_id}: Empty transcript, skipping")
            except Exception as e:
                print(f"Transcription worker error: {e}")
            finally:
                await self._send_metrics(websocket, stt={
                    "status": "completed", "segment_id": request.segment_id, "latency": stt_latency
                })
                self.transcription_queue.task_done()

    async def process_llm_and_tts(self, websocket: WebSocket):
        loop = asyncio.get_event_loop()
        pending_tasks: List[asyncio.Task] = []
        assistant_entry = None
        cancelled = errored = False

        try:
            pending = self.transcription_queue.qsize()
            if pending:
                print(f"[llm] Waiting for {pending} pending transcription job(s)")
            await self.transcription_queue.join()

            user_utterances = [m for m in self.conversation_history if m["role"] == "user"]
            if not user_utterances:
                print("No user input to respond to")
                return

            self.is_playing_response = True
            await websocket.send_text(json.dumps({
                "event": "state", "state": self.state.value, "vad": self.smoothed_prob,
                "segments": self.segment_count, "listening": self.is_listening, "responding": True
            }))

            print(f"[llm] Generating response for: {user_utterances[-1]['content']}")
            llm_start = time.monotonic()
            try:
                llm_response = await loop.run_in_executor(None, self.llm_handler.generate_response, self.conversation_history)
            except Exception as llm_error:
                print(f"LLM generation failed: {llm_error}")
                llm_response = "I apologize, I'm having trouble processing that right now."
            llm_first_latency = time.monotonic() - llm_start

            llm_response = (llm_response or "").strip()
            if not llm_response:
                print("Empty LLM response, skipping")
                return

            sentences = _split_sentences(llm_response) or [llm_response]
            assistant_entry = {"role": "assistant", "content": " ".join(sentences).strip()}
            self.conversation_history.append(assistant_entry)
            print(f"[llm] Streaming {len(sentences)} sentence(s)")

            previous_tts: Optional[asyncio.Task] = None
            tts_start = tts_first_latency = None

            for sentence in sentences:
                # Check if interrupted during streaming
                if cancelled:
                    break
                    
                print(f"[llm] Sentence: {sentence}")
                await websocket.send_text(json.dumps({"event": "text", "role": "assistant", "text": sentence}))
                if tts_start is None:
                    tts_start = time.monotonic()
                tts_task = asyncio.create_task(asyncio.to_thread(self.tts_handler.generate_speech, sentence))
                pending_tasks.append(tts_task)
                if previous_tts:
                    audio_bytes = await previous_tts
                    if audio_bytes and not cancelled:
                        if tts_first_latency is None and tts_start is not None:
                            tts_first_latency = time.monotonic() - tts_start
                        print(f"[tts] Audio chunk {len(audio_bytes)} bytes")
                        await websocket.send_text(json.dumps({
                            "event": "media", "mime": "audio/wav", "audio": _encode_bytes(audio_bytes)
                        }))
                previous_tts = tts_task

            if previous_tts and not cancelled:
                audio_bytes = await previous_tts
                if audio_bytes:
                    if tts_first_latency is None and tts_start is not None:
                        tts_first_latency = time.monotonic() - tts_start
                    print(f"[tts] Audio chunk {len(audio_bytes)} bytes")
                    await websocket.send_text(json.dumps({
                        "event": "media", "mime": "audio/wav", "audio": _encode_bytes(audio_bytes)
                    }))

            if not cancelled:
                await self._send_metrics(websocket, llm={"first_token": llm_first_latency}, tts={"first_audio": tts_first_latency})
                await websocket.send_text(json.dumps({
                    "event": "text", "role": "assistant", "text": llm_response, "complete": True
                }))
        except asyncio.CancelledError:
            cancelled = True
            print("LLM/TTS processing cancelled by interruption")
            raise
        except Exception as exc:
            errored = True
            print(f"Error in LLM/TTS processing: {exc}")
        finally:
            for task in pending_tasks:
                if not task.done():
                    task.cancel()
            if (cancelled or errored) and assistant_entry and assistant_entry in self.conversation_history:
                try:
                    self.conversation_history.remove(assistant_entry)
                    print("[interrupt] Removed incomplete assistant response from history")
                except ValueError:
                    pass
            if self.is_playing_response:
                self.is_playing_response = False
                await websocket.send_text(json.dumps({
                    "event": "state", "state": self.state.value, "vad": self.smoothed_prob,
                    "segments": self.segment_count, "listening": self.is_listening, "responding": False
                }))

    async def _send_metrics(self, websocket: WebSocket, stt: Optional[Dict] = None, llm: Optional[Dict] = None, tts: Optional[Dict] = None):
        payload: Dict[str, Dict] = {}
        if stt is not None:
            payload["stt"] = stt
        if llm is not None:
            payload["llm"] = llm
        if tts is not None:
            payload["tts"] = tts
        if payload:
            await websocket.send_text(json.dumps({"event": "metrics", "metrics": payload}))
    
    def _get_vad_probability(self, chunk):
        x = np.concatenate([self._vad_context, chunk.reshape(1, -1)], axis=1)
        out, self._vad_state = self.vad_session.run(
            None, {'input': x, 'state': self._vad_state, 'sr': np.array([16000], dtype=np.int64)}
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


class VoicePipelineOrchestrator:
    def __init__(self, websocket: WebSocket, resources: PipelineResources):
        self.websocket = websocket
        self.resources = resources
        self.detector = SpeechDetector(resources)
        self._transcription_worker: Optional[asyncio.Task] = None
        self._response_task: Optional[asyncio.Task] = None

    async def start(self):
        self._transcription_worker = asyncio.create_task(self.detector.process_transcription_queue(self.websocket))
        await self._send_state()

    async def shutdown(self):
        await self._cancel_task(self._response_task)
        await self._cancel_task(self._transcription_worker)
        self._response_task = None
        self._transcription_worker = None

    async def handle_event(self, payload: dict):
        event_type = payload.get("event")

        if event_type == "start":
            self.detector.start_listening()
            print("[event] Listening started via client")
            await self._send_state()
        elif event_type == "stop":
            if payload.get("target") == "playback":
                await self._interrupt_response()
            else:
                self.detector.stop_listening()
                print("[event] Listening stopped via client")
                await self._send_state()
        elif event_type == "media":
            audio_encoded = payload.get("audio")
            if audio_encoded:
                audio_array = _decode_float32_chunk(audio_encoded)
                if audio_array is not None and len(audio_array) > 0:
                    frames = len(audio_array) // 512
                    for idx in range(frames):
                        chunk = audio_array[idx * 512:(idx + 1) * 512]
                        events = self.detector.process_chunk(chunk)
                        if events:
                            await self._dispatch_events(events)
                    await self._send_state()
        elif event_type == "interrupt":
            await self._interrupt_response()
            await self._send_state()

    async def _dispatch_events(self, events: List[PipelineEvent]):
        should_interrupt = (
            PipelineEvent.SPEECH_STARTING in events and
            (self.detector.is_playing_response or (self._response_task and not self._response_task.done()))
        )
        if should_interrupt:
            interrupted = await self._interrupt_response()
            if interrupted:
                await self.websocket.send_text(json.dumps({"event": "interrupt"}))

        for event in events:
            if event == PipelineEvent.START_TRANSCRIPTION:
                await self.detector.queue_transcription()
            elif event == PipelineEvent.PROCESS_RESPONSE:
                await self._ensure_response_task()
            elif event == PipelineEvent.SPEECH_STARTING:
                await self.websocket.send_text(json.dumps({"event": "start", "state": "started"}))
            elif event == PipelineEvent.SPEECH_STARTED:
                await self.websocket.send_text(json.dumps({"event": "start", "state": "speaking"}))
            elif event == PipelineEvent.SPEECH_STOPPING:
                await self.websocket.send_text(json.dumps({"event": "stop", "state": "stop"}))
            elif event == PipelineEvent.SPEECH_ENDED:
                await self.websocket.send_text(json.dumps({"event": "stop", "state": "quiet"}))
            elif event == PipelineEvent.SPEECH_RESUMED:
                await self.websocket.send_text(json.dumps({"event": "start", "state": "speaking"}))
        await self._send_state()

    async def _interrupt_response(self) -> bool:
        interrupted = False
        if self._response_task and not self._response_task.done():
            await self._cancel_task(self._response_task)
            interrupted = True
        self._response_task = None
        if self.detector.is_playing_response:
            self.detector.is_playing_response = False
            interrupted = True
        return interrupted

    async def _ensure_response_task(self):
        if self._response_task and not self._response_task.done():
            return
        self._response_task = asyncio.create_task(self.detector.process_llm_and_tts(self.websocket))
        self._response_task.add_done_callback(self._log_task_exception)

    async def _cancel_task(self, task: Optional[asyncio.Task]):
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    def _log_task_exception(self, task: asyncio.Task):
        if task.cancelled():
            if task is self._response_task:
                self._response_task = None
            return
        exc = task.exception()
        if exc:
            print(f"Background task error: {exc}")
        if task is self._response_task:
            self._response_task = None

    async def _send_state(self):
        snapshot = self.detector.get_current_state()
        await self.websocket.send_text(json.dumps({
            "event": "state", "state": snapshot["state"], "vad": snapshot["smoothed_prob"],
            "segments": snapshot["segments_count"], "listening": snapshot["is_listening"],
            "responding": snapshot["is_playing_response"]
        }))


PIPELINE_RESOURCES = PipelineResources()
app = FastAPI()


@app.get("/")
async def get():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/correlator.js")
async def get_correlator():
    with open("correlator.js", "r") as f:
        from fastapi.responses import Response
        return Response(content=f.read(), media_type="application/javascript")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    orchestrator = VoicePipelineOrchestrator(websocket, PIPELINE_RESOURCES)
    await orchestrator.start()
    try:
        while True:
            message_text = await websocket.receive_text()
            await orchestrator.handle_event(json.loads(message_text))
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await orchestrator.shutdown()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)