# Real-Time Speech Detection & Transcription

Real-time speech-to-text system using VAD, End-of-Utterance detection, and MLX Whisper.

## Demo

[![Watch the demo](https://img.youtube.com/vi/uBqNq66qUos/maxresdefault.jpg)](https://www.youtube.com/watch?v=uBqNq66qUos)

## Features

- Real-time voice activity detection (Silero VAD)
- Smart speech segmentation with safety buffers
- End-of-utterance detection
- Fast transcription with MLX Whisper (Apple Silicon)
- Queue-based async processing
- Clean web interface

## Installation

```bash
git clone https://github.com/user/repo.git
cd repo
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

ONNX models files are in models folder:
- `silero_vad.onnx` - Silero VAD V6
- `smart_turn_v3.onnx` - EoU model (by pipecat)

## Usage

```bash
python server.py
```

Open `http://localhost:8000` and click "Start Listening".

## Configuration

Edit `server.py`:

```python
ALPHA = 0.2              # VAD smoothing
START_THRESH = 0.3       # Speech start threshold
SPEAK_THRESH = 0.5       # Speaking threshold
STOP_THRESH = 0.3        # Speech stop threshold
QUIET_THRESH = 0.05      # Quiet threshold
SAFETY_BEFORE = 4        # Pre-buffer chunks (128ms)
SAFETY_AFTER = 2         # Post-buffer chunks (64ms)
MAX_QUEUE_SIZE = 2       # Max pending transcriptions
```

## Architecture

```
Audio (16kHz) → VAD → State Machine → Buffer → EoU → Queue → Whisper → UI
```

**States**: QUIET → STARTING → SPEAKING → STOPPING

## Project Structure

```
├── server.py           # FastAPI server + VAD + state machine
├── transcriber.py      # MLX Whisper wrapper
├── index.html          # Web UI
├── requirements.txt    # Dependencies
└── assets/
    └── demo.mp4       # Demo video
```

## Requirements

- Python 3.10+
- macOS with Apple Silicon
- Microphone access