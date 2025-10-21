# Technical Overview

Fast snapshot of how the <1 s MLX voice agent stays responsive.

## Click below for demo
[![Watch the demo](https://img.youtube.com/vi/6IEK2fXB_ok/0.jpg)](https://www.youtube.com/watch?v=6IEK2fXB_ok)

## Pipeline
1. **Browser loop**
   - `correlator.js` AudioWorklet fuses mic + TTS reference; flags echo vs speech.
   - Rolling buffer keeps ~500 ms of context. Interruption bursts instantly trigger `stop` → server.
2. **Server orchestration (`VoicePipeline`)**
   - `SpeechDetector` runs Silero VAD + optional SmartTurn EoU; emits `TRANSCRIBE` / `RESPOND`.
   - Shared `asyncio.Lock` guards MLX so Whisper and the LLM never overlap.
   - A per-response `asyncio.Event` cancels LLM streaming + Kokoro generation the moment barge-in occurs.
3. **Models**
   - Whisper (MLX) transcribes queued segments in a background worker.
   - LLM streams sentence chunks; each chunk feeds Kokoro synchronously to keep audio ordered.
   - Completed turns update conversation history; partial turns are dropped on cancellation.

## Key Numbers
- Audio chunks: 512 samples (32 ms @ 16 kHz)
- Interruption detection: ~64 ms (2 frames) including playback halt
- Whisper latency: 0.3–0.7 s for typical user turns
- First LLM token: ~250 ms; first TTS audio: ~200 ms
- End-to-end (speech end → bot audio): **<1 s** steady-state

## Files of Interest
- `server.py` — event loop, cancellation, metrics
- `audio_buffer.py` — pre-buffered segment capture
- `index.html` — UI, interruption thresholds, playback queue

That’s the whole story: tight buffers, immediate cancellation, and MLX-only workloads keep everything under a second.
