# server.py
"""Optimized voice agent server with MLX concurrency protection"""
import json
import base64
import asyncio
import time
from datetime import datetime
from typing import Optional, List, Dict
from enum import Enum

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response

from config import *
from audio_buffer import AudioBuffer, SpeechState, save_audio_to_wav, split_audio_into_chunks
from vad_detector import VADDetector, EndOfUtteranceDetector
from llm_handler import LLMHandler
from transcriber import RealtimeTranscriber
from tts_handler import TTSHandler


# =============================================================================
# Utilities
# =============================================================================

def encode_audio(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def decode_float32_audio(data: str) -> Optional[np.ndarray]:
    try:
        audio_bytes = base64.b64decode(data.encode("ascii"))
        return np.frombuffer(audio_bytes, dtype=np.float32) if len(audio_bytes) % 4 == 0 else None
    except:
        return None


# =============================================================================
# Events
# =============================================================================

class PipelineEvent(str, Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"
    TRANSCRIBE = "transcribe"
    RESPOND = "respond"


# =============================================================================
# Resources (Singleton)
# =============================================================================

class PipelineResources:
    """Global resources initialized once"""
    
    def __init__(self):
        print("Initializing pipeline...")
        
        self.transcriber = RealtimeTranscriber() if ENABLE_TRANSCRIPTION else None
        self.llm_handler = LLMHandler()
        self.tts_handler = TTSHandler()
        
        # CRITICAL: Global lock for MLX operations (Whisper + LLM share MLX runtime)
        # This prevents heap corruption from concurrent MLX access
        self.mlx_lock = asyncio.Lock()
        
        # Warm up transcriber
        if self.transcriber:
            dummy_audio = np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.001
            try:
                self.transcriber.transcribe(dummy_audio)
                print("Transcriber warmed up")
            except Exception as e:
                print(f"Warmup warning: {e}")
        
        print("Pipeline ready\n")


# =============================================================================
# Speech Detector
# =============================================================================

class SpeechDetector:
    """Manages VAD state machine and audio segmentation"""
    
    def __init__(self):
        self.vad = VADDetector()
        self.eou = EndOfUtteranceDetector() if ENABLE_TRANSCRIPTION else None
        self.buffer = AudioBuffer()
        
        self.state = SpeechState.QUIET
        self.is_listening = False
        self.is_responding = False
        self.user_speaking = False  # Tracks if user is mid-utterance (across pauses)
        
        self.recording: List[np.ndarray] = []
        self.segment_count = 0
        self.current_segment: Optional[np.ndarray] = None
    
    def start_listening(self):
        self.is_listening = True
        self.recording = []
        self.segment_count = 0
        self.user_speaking = False
        print("[detector] Started")
    
    def stop_listening(self):
        self.is_listening = False
        self.user_speaking = False
        
        if ENABLE_RECORDING and self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_audio_to_wav(np.concatenate(self.recording), f"recording_{timestamp}.wav")
            print(f"[detector] Saved recording")
        
        print(f"[detector] Stopped (segments: {self.segment_count})\n")
    
    def process_chunk(self, chunk: np.ndarray) -> List[PipelineEvent]:
        if not self.is_listening:
            return []
        
        if ENABLE_RECORDING:
            self.recording.append(chunk.copy())
        
        vad_prob = self.vad.process_chunk(chunk)
        self.buffer.add_chunk(chunk, self.state)
        
        # Always feed EOU detector so it has continuous context
        if self.eou:
            self.eou.add_audio(chunk)
        
        return self._update_state(vad_prob)
    
    def _update_state(self, vad_prob: float) -> List[PipelineEvent]:
        events = []
        prev_state = self.state
        
        if self.state == SpeechState.QUIET:
            if vad_prob >= VAD_START_THRESHOLD:
                self.state = SpeechState.STARTING
                self.user_speaking = True  # User started speaking
                events.append(PipelineEvent.SPEECH_START)
        
        elif self.state == SpeechState.STARTING:
            if vad_prob >= VAD_SPEAKING_THRESHOLD:
                self.state = SpeechState.SPEAKING
            elif vad_prob < VAD_QUIET_THRESHOLD:
                self.state = SpeechState.QUIET
                self.user_speaking = False  # False start
        
        elif self.state == SpeechState.SPEAKING:
            if vad_prob < VAD_STOP_THRESHOLD:
                self.state = SpeechState.STOPPING
                # Capture segment for transcription
                self.current_segment = self.buffer.get_segment()
                if self.current_segment is not None:
                    print(f"[detector] Segment ({len(self.current_segment)/SAMPLE_RATE:.2f}s)")
                    events.append(PipelineEvent.TRANSCRIBE)
        
        elif self.state == SpeechState.STOPPING:
            vad_quiet = vad_prob < VAD_QUIET_THRESHOLD
            
            eou_confirms = not self.eou
            if self.eou and vad_quiet and self.eou.has_enough_audio():
                result = self.eou.detect()
                eou_confirms = result['ended'] and result['confidence'] > EOU_CONFIDENCE_THRESHOLD
                if eou_confirms:
                    print(f"[detector] EOU (conf: {result['confidence']:.2f})")
            
            if vad_quiet and eou_confirms:
                self.state = SpeechState.QUIET
                events.append(PipelineEvent.SPEECH_END)
                
                if self.user_speaking:
                    events.append(PipelineEvent.RESPOND)
                    self.user_speaking = False
                
                if self.eou:
                    self.eou.reset()
                self.current_segment = None
            
            # User resumed speaking (just a pause)
            elif vad_prob > VAD_SPEAKING_THRESHOLD:
                self.state = SpeechState.SPEAKING
                self.current_segment = None
                # user_speaking stays True
        
        if prev_state != self.state:
            print(f"[state] {prev_state.value} → {self.state.value} (vad: {vad_prob:.3f})")
        
        return events
    
    def get_state(self) -> dict:
        return {
            'state': self.state.value,
            'vad_prob': float(self.vad.smoothed_prob),
            'segments': self.segment_count,
            'listening': self.is_listening,
            'responding': self.is_responding
        }


# =============================================================================
# Voice Pipeline Orchestrator
# =============================================================================

class VoicePipeline:
    """Orchestrates the complete voice interaction pipeline"""
    
    def __init__(self, ws: WebSocket, resources: PipelineResources):
        self.ws = ws
        self.resources = resources
        self.detector = SpeechDetector()
        
        self.conversation: List[Dict[str, str]] = []
        self.transcription_queue = asyncio.Queue(maxsize=MAX_TRANSCRIPTION_QUEUE_SIZE)
        self.accumulated_text = ""
        self.is_accumulating = False
        
        self._transcription_task: Optional[asyncio.Task] = None
        self._response_task: Optional[asyncio.Task] = None
        self._response_lock = asyncio.Lock()
        self._response_cancel_event: Optional[asyncio.Event] = None
    
    async def start(self):
        self._transcription_task = asyncio.create_task(self._transcription_worker())
        await self._send_state()
        print("[pipeline] Started\n")
    
    async def shutdown(self):
        if self._response_cancel_event and not self._response_cancel_event.is_set():
            self._response_cancel_event.set()
        
        await self._cancel_response_task()
        if self._response_cancel_event and self._response_cancel_event.is_set():
            self._response_cancel_event = None
        
        if self._transcription_task:
            self._transcription_task.cancel()
            try:
                await self._transcription_task
            except asyncio.CancelledError:
                pass
            self._transcription_task = None
        
        print("[pipeline] Shutdown\n")
    
    async def handle_message(self, payload: dict):
        event = payload.get("event")
        
        if event == "start":
            self.detector.start_listening()
            await self._send_state()
        
        elif event == "stop":
            if payload.get("target") == "playback":
                await self._interrupt_response(notify_client=True)
            else:
                self.detector.stop_listening()
            await self._send_state()
        
        elif event == "media":
            await self._handle_audio(payload.get("audio"))
        
        elif event == "interrupt":
            await self._interrupt_response(notify_client=True)
            await self._send_state()
    
    async def _handle_audio(self, audio_data: str):
        if not audio_data:
            return
        
        audio = decode_float32_audio(audio_data)
        if audio is None or len(audio) == 0:
            return
        
        for chunk in split_audio_into_chunks(audio, CHUNK_SIZE):
            events = self.detector.process_chunk(chunk)
            if events:
                await self._handle_events(events)
        
        await self._send_state()
    
    async def _handle_events(self, events: List[PipelineEvent]):
        if PipelineEvent.SPEECH_START in events:
            self.is_accumulating = True
            self.accumulated_text = ""
            print("[pipeline] Accumulating started")
            
            # Interrupt any ongoing response
            if self.detector.is_responding or (self._response_task and not self._response_task.done()):
                await self._interrupt_response(notify_client=True)
        
        for event in events:
            if event == PipelineEvent.TRANSCRIBE:
                await self._queue_transcription()
            
            elif event == PipelineEvent.RESPOND:
                await self._finalize_and_respond()
            
            elif event in [PipelineEvent.SPEECH_START, PipelineEvent.SPEECH_END]:
                await self.ws.send_text(json.dumps({"event": event.value}))
        
        await self._send_state()
    
    async def _queue_transcription(self):
        if not self.resources.transcriber or self.detector.current_segment is None:
            return
        
        segment = self.detector.current_segment
        if len(segment) < SAMPLE_RATE * MIN_SEGMENT_DURATION:
            return
        
        self.detector.segment_count += 1
        segment_id = self.detector.segment_count
        
        try:
            self.transcription_queue.put_nowait((segment_id, segment))
            print(f"[transcribe] Queued #{segment_id}")
        except asyncio.QueueFull:
            print(f"[transcribe] Queue full, dropped #{segment_id}")
    
    async def _transcription_worker(self):
        """Background worker for transcription with MLX lock protection"""
        if not self.resources.transcriber:
            return
        
        print("[transcribe] Worker started")
        
        while True:
            segment_id, audio = await self.transcription_queue.get()
            
            try:
                # Send STT status
                await self._send_metrics(stt={"status": "running", "segment_id": segment_id})
                
                # CRITICAL: Acquire MLX lock before transcription
                # This prevents concurrent Whisper + LLM access to MLX runtime
                stt_start = time.monotonic()
                async with self.resources.mlx_lock:
                    transcript = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.resources.transcriber.transcribe,
                        audio
                    )
                stt_latency = time.monotonic() - stt_start
                
                # Send STT completion
                await self._send_metrics(stt={"status": "completed", "segment_id": segment_id, "latency": stt_latency})
                
                if transcript and transcript.strip():
                    print(f"[transcribe] #{segment_id}: {transcript} ({stt_latency:.2f}s)")
                    
                    if self.is_accumulating:
                        # Accumulate transcript across pauses
                        if self.accumulated_text:
                            self.accumulated_text += " " + transcript.strip()
                        else:
                            self.accumulated_text = transcript.strip()
                        
                        print(f"[transcribe] Accumulated: {self.accumulated_text}")
                        
                        # Send partial transcript to client
                        await self.ws.send_text(json.dumps({
                            "event": "text",
                            "role": "user",
                            "text": transcript.strip(),
                            "segment_id": segment_id,
                            "partial": True
                        }))
            
            except Exception as e:
                print(f"[transcribe] Error: {e}")
                import traceback
                traceback.print_exc()
            
            finally:
                self.transcription_queue.task_done()
    
    async def _finalize_and_respond(self):
        """Finalize accumulated transcript and generate response"""
        # Wait for all pending transcriptions to complete
        if self.transcription_queue.qsize() > 0:
            print(f"[pipeline] Waiting for {self.transcription_queue.qsize()} transcriptions")
            await self.transcription_queue.join()
        
        self.is_accumulating = False
        
        if not self.accumulated_text:
            print("[pipeline] No text to respond to")
            return
        
        final_text = self.accumulated_text.strip()
        print(f"[pipeline] Final: {final_text}")
        
        # Add to conversation history (only once!)
        self.conversation.append({"role": "user", "content": final_text})
        
        # Send complete transcript to client
        await self.ws.send_text(json.dumps({
            "event": "text",
            "role": "user",
            "text": final_text,
            "complete": True
        }))
        
        self.accumulated_text = ""
        
        # Generate response
        await self._start_response()
    
    async def _start_response(self):
        async with self._response_lock:
            # Cancel any existing response task before starting a new one
            await self._cancel_response_task()
            
            # Fresh cancellation event for the new response
            self._response_cancel_event = asyncio.Event()
            self._response_task = asyncio.create_task(
                self._generate_response(self._response_cancel_event)
            )

    async def _cancel_response_task(self) -> bool:
        """Cancel the active response task if it exists."""
        task = self._response_task
        if not task:
            return False
        
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._response_task = None
        return True
    
    async def _generate_response(self, cancel_event: asyncio.Event):
        """Generate LLM response with streaming TTS.
        
        cancel_event: signals when downstream components should abandon work.
        """
        try:
            user_msgs = [m for m in self.conversation if m["role"] == "user"]
            if not user_msgs:
                return
            
            if cancel_event.is_set():
                return
            
            self.detector.is_responding = True
            await self._send_state()
            
            print(f"[response] Generating for: {user_msgs[-1]['content']}")
            
            full_response = []
            llm_start = time.monotonic()
            tts_start = None
            first_llm_time = None
            first_tts_time = None
            idx = 0
            
            # CRITICAL: Acquire MLX lock for entire LLM generation
            # This prevents concurrent Whisper transcription during LLM use
            async with self.resources.mlx_lock:
                # for sentence in self.resources.llm_handler.stream_response_batched(self.conversation):
                for sentence in self.resources.llm_handler.stream_response(self.conversation):
                    if cancel_event.is_set() or not self.detector.is_responding:
                        print("[response] Interrupted")
                        break
                    
                    full_response.append(sentence)
                    
                    if first_llm_time is None:
                        first_llm_time = time.monotonic() - llm_start
                        await self._send_metrics(llm={"first_token": first_llm_time})
                    
                    # Send sentence to client
                    await self.ws.send_text(json.dumps({
                        "event": "text",
                        "role": "assistant",
                        "text": sentence
                    }))
                    
                    # Start TTS timing
                    if tts_start is None:
                        tts_start = time.monotonic()
                    
                    # Generate TTS and await it (sequential for proper order)
                    await self._send_tts(sentence, idx, cancel_event)
                    
                    # Record first TTS latency
                    if first_tts_time is None and tts_start is not None:
                        first_tts_time = time.monotonic() - tts_start
                        await self._send_metrics(tts={"first_audio": first_tts_time})
                    
                    idx += 1
            
            # Add complete response to history
            complete = " ".join(full_response).strip()
            if complete and self.detector.is_responding and not cancel_event.is_set():
                self.conversation.append({"role": "assistant", "content": complete})
                
                await self.ws.send_text(json.dumps({
                    "event": "text",
                    "role": "assistant",
                    "text": complete,
                    "complete": True
                }))
                
                print(f"[response] Complete ({first_llm_time:.2f}s to first)")
        
        except asyncio.CancelledError:
            print("[response] Cancelled")
            raise
        
        except Exception as e:
            print(f"[response] Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.detector.is_responding = False
            await self._send_state()
            if self._response_cancel_event is cancel_event:
                self._response_cancel_event = None
    
    async def _send_tts(self, text: str, index: int, cancel_event: asyncio.Event):
        """Generate and send TTS audio"""
        try:
            if cancel_event.is_set():
                return
            
            audio_bytes = await asyncio.get_event_loop().run_in_executor(
                None,
                self.resources.tts_handler.generate_speech,
                text
            )
            
            if cancel_event.is_set() or not self.detector.is_responding:
                return
            
            if audio_bytes:
                await self.ws.send_text(json.dumps({
                    "event": "media",
                    "mime": "audio/wav",
                    "audio": encode_audio(audio_bytes),
                    "index": index
                }))
                print(f"[tts] Sent {len(audio_bytes)} bytes (#{index})")
        
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[tts] Error: {e}")
    
    async def _send_metrics(self, stt: Optional[Dict] = None, llm: Optional[Dict] = None, tts: Optional[Dict] = None):
        """Send performance metrics to client"""
        payload = {}
        if stt is not None:
            payload["stt"] = stt
        if llm is not None:
            payload["llm"] = llm
        if tts is not None:
            payload["tts"] = tts
        
        if payload:
            await self.ws.send_text(json.dumps({
                "event": "metrics",
                "metrics": payload
            }))
    
    async def _interrupt_response(self, notify_client: bool = False) -> bool:
        """Interrupt ongoing response generation and optionally notify the client."""
        interrupted = False
        
        cancel_event = self._response_cancel_event
        if cancel_event and not cancel_event.is_set():
            cancel_event.set()
            interrupted = True
        
        async with self._response_lock:
            task_cancelled = await self._cancel_response_task()
            if self._response_cancel_event and self._response_cancel_event.is_set() and not self._response_task:
                self._response_cancel_event = None
        
        if task_cancelled:
            interrupted = True
        
        if self.detector.is_responding:
            self.detector.is_responding = False
            interrupted = True
        
        if interrupted:
            if self.conversation and self.conversation[-1]["role"] == "assistant":
                self.conversation.pop()
                print("[interrupt] Removed incomplete response")
            
            if notify_client:
                await self.ws.send_text(json.dumps({"event": "interrupt"}))
        
        return interrupted
    
    async def _send_state(self):
        """Send current state to client"""
        await self.ws.send_text(json.dumps({
            "event": "state",
            **self.detector.get_state()
        }))


# =============================================================================
# FastAPI Application
# =============================================================================

RESOURCES = PipelineResources()
app = FastAPI()


@app.get("/")
async def get_index():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/correlator.js")
async def get_correlator():
    with open("correlator.js", "r") as f:
        return Response(content=f.read(), media_type="application/javascript")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[ws] Client connected")
    
    pipeline = VoicePipeline(websocket, RESOURCES)
    await pipeline.start()
    
    try:
        while True:
            message = await websocket.receive_text()
            await pipeline.handle_message(json.loads(message))
    
    except WebSocketDisconnect:
        print("[ws] Client disconnected")
    
    except Exception as e:
        print(f"[ws] Error: {e}")
    
    finally:
        await pipeline.shutdown()


if __name__ == "__main__":
    import uvicorn
    print("Starting voice agent server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
