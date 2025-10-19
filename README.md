# Real-Time Voice Agent with Barge-In

Full-duplex voice assistant with real-time speech detection, transcription, LLM response generation, and text-to-speech. Features professional-grade barge-in using reference signal correlation for echo cancellation.

## Demo

[![Watch the demo](https://img.youtube.com/vi/uBqNq66qUos/0.jpg)](https://www.youtube.com/watch?v=uBqNq66qUos)

## Features

### Core Pipeline
- **Voice Activity Detection (VAD)**: Silero VAD with exponential smoothing
- **Speech Segmentation**: Smart buffering with safety margins (pre/post-buffer)
- **End-of-Utterance Detection**: ML-based turn-taking detection
- **Speech-to-Text**: MLX Whisper for Apple Silicon (fast, on-device)
- **LLM Integration**: Conversational response generation with history
- **Text-to-Speech**: Kokoro TTS for natural voice synthesis

### Advanced Features
- **Full-Duplex Barge-In**: Interrupt bot responses naturally
- **Echo Cancellation**: Reference signal correlation (AudioWorklet-based)
- **Async Processing**: Queue-based STT with parallel LLM/TTS streaming
- **Rolling Audio Buffer**: Captures pre-interruption context (~500ms)
- **WebSocket Communication**: Real-time bidirectional audio/text streaming
- **Responsive UI**: Live state visualization and conversation history

## Installation

### Prerequisites
- Python 3.10+
- macOS with Apple Silicon (for MLX)
- Microphone access

### Setup

```bash
git clone https://github.com/user/repo.git
cd repo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Required Models

Place these ONNX models in the `models/` folder:
- `silero_vad.onnx` - Silero VAD V6 (voice activity detection)
- `smart_turn_v3.onnx` - End-of-utterance model (by Pipecat)

Models are included in the repository.

## Usage

```bash
python server.py
```

Open `http://localhost:8000` in your browser and click **"Start Listening"**.

### How to Use
1. Click "Start Listening" to activate microphone
2. Speak naturally - the bot will detect when you're done speaking
3. Bot responds with voice - you can interrupt anytime by speaking
4. Conversation history is maintained throughout the session

## Configuration

Edit thresholds in `server.py`:

```python
# VAD Configuration
ALPHA = 0.2                      # Smoothing factor (0-1)
START_THRESHOLD = 0.3            # Speech start detection
SPEAKING_THRESHOLD = 0.5         # Confirmed speaking
STOP_THRESHOLD = 0.3             # Speech stopping
QUIET_THRESHOLD = 0.05           # Return to quiet

# Buffer Configuration
SAFETY_CHUNKS_BEFORE = 4         # Pre-buffer: 128ms (4 × 32ms)

# Queue Configuration
MAX_TRANSCRIPTION_QUEUE_SIZE = 2 # Max pending transcriptions

# Feature Flags
ENABLE_RECORDING = False         # Save full audio to WAV
ENABLE_TRANSCRIPTION = True      # Enable STT processing
```

Edit interruption sensitivity in `index.html`:

```javascript
const INTERRUPTION = {
    RMS: 0.012,           // Minimum RMS threshold
    EXTRA: 0.008,         // Extra margin above average
    RATIO: 1.20,          // Ratio above baseline
    ABS: 0.035,           // Absolute threshold
    MIN_FRAMES: 2,        // Frames before checking
    REQUIRED_FRAMES: 2    // Frames needed to trigger
};
```

## Architecture

### High-Level Pipeline

```
Microphone (16kHz)
    ↓
Correlator (Echo Cancellation)
    ↓
VAD (Silero) → State Machine (QUIET/STARTING/SPEAKING/STOPPING)
    ↓
Speech Segmentation Buffer (with safety margins)
    ↓
End-of-Utterance Detector (optional)
    ↓
Transcription Queue (async, max 2 pending)
    ↓
STT (MLX Whisper) → Text
    ↓
LLM (MLX LLM) → Response
    ↓
TTS (Kokoro) → Audio
    ↓
Speakers → [Reference signal back to Correlator]
```

### State Machine

```
QUIET ──(VAD ≥ 0.3)──→ STARTING ──(VAD ≥ 0.5)──→ SPEAKING
  ↑                                                   ↓
  │                                              (VAD < 0.3)
  │                                                   ↓
  └──────(timeout)──────── STOPPING ←───(VAD ≥ 0.5)──┘
                              ↓
                        (EoU detected or
                         safety margin met)
                              ↓
                         Send to STT
```

### Barge-In Architecture

```
Bot Audio (TTS) ──→ Correlator (Input 1: Reference)
                         ↓
Microphone ──→ Correlator (Input 0: Mic)
                         ↓
              Correlation Analysis
              (Cosine Similarity)
                         ↓
         ┌───────────────┴───────────────┐
         ↓                               ↓
   High Correlation              Low Correlation
   (corr > 0.30)                (corr < 0.30)
   = Bot Echo                    = User Speech
         ↓                               ↓
   Block from VAD              Energy Spike Detection
                                        ↓
                                Interruption Triggered
                                        ↓
                              Stop Playback + Send Audio
```

## Project Structure

```
├── server.py              # FastAPI server, VAD, state machine, orchestration
├── transcriber.py         # MLX Whisper wrapper for STT
├── llm_handler.py         # MLX LLM integration for response generation
├── tts_handler.py         # Kokoro TTS for speech synthesis
├── index.html             # Web UI with barge-in logic
├── correlator.js          # AudioWorklet for echo cancellation
├── requirements.txt       # Python dependencies
└── models/
    ├── silero_vad.onnx    # VAD model
    └── smart_turn_v3.onnx # End-of-utterance model
```

## Technical Details

### Audio Processing
- Sample rate: 16 kHz (mic) / 24 kHz (TTS)
- Frame size: 512 samples (32ms @ 16kHz)
- Buffer: 30 frames rolling window (~1 second)
- Latency: ~64ms interruption detection

### Echo Cancellation
- Method: Reference signal correlation (AudioWorklet)
- Correlation threshold: >0.30 = echo
- RMS thresholds: Dynamic adaptive baseline
- Updates: Every 32ms (per audio frame)

### Models Used
- **VAD**: Silero VAD v6 (ONNX)
- **EoU**: Smart Turn v3 (Whisper-based, ONNX)
- **STT**: Whisper Small EN (MLX quantized)
- **LLM**: LFM2-1.2B-4bit (MLX quantized)
- **TTS**: Kokoro-82M (PyTorch)

## Performance

- **Interruption latency**: <100ms (typically 64ms)
- **STT latency**: ~0.5-2s (depends on audio length)
- **LLM latency**: ~0.3-1s first token
- **TTS latency**: ~0.2-0.5s first audio
- **End-to-end**: ~1-3s from speech end to response start

## Requirements

### Python Packages
```
fastapi
uvicorn[standard]
websockets
numpy
onnxruntime
transformers
mlx-whisper
mlx-lm
kokoro
torch
```

See `requirements.txt` for exact versions.

### Hardware
- Apple Silicon Mac (M1/M2/M3)
- Minimum 8GB RAM
- Microphone and speakers

## Troubleshooting

### Echo/Feedback Issues
1. Use headphones (eliminates echo completely)
2. Reduce speaker volume
3. Ensure `echoCancellation: true` in getUserMedia
4. Check correlation values in browser console

### Interruption Not Working
1. Check browser console for `[interrupt]` logs
2. Verify correlator.js is loaded (check Network tab)
3. Speak louder/closer to microphone
4. Adjust `INTERRUPTION` thresholds in index.html

### Missing First Words
1. Check `audioBuffer.getRecent(15)` setting
2. Increase pre-buffer frames if needed
3. Verify interruption detection latency

### Transcription Errors
1. Reduce background noise
2. Speak clearly and at normal volume
3. Check `no_speech_threshold` in transcriber.py
4. Ensure minimum audio length (>0.3s)

## License

MIT

## Credits

- Silero VAD: [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- Whisper: [openai/whisper](https://github.com/openai/whisper)
- MLX: [ml-explore/mlx](https://github.com/ml-explore/mlx)
- Kokoro TTS: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- Smart Turn: [pipecat-ai](https://github.com/pipecat-ai/pipecat)
