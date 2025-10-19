# config.py
"""Configuration constants for the voice agent"""

# Audio Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 512  # 32ms at 16kHz
CHANNELS = 1

# VAD Configuration
VAD_ALPHA = 0.1
VAD_START_THRESHOLD = 0.3
VAD_SPEAKING_THRESHOLD = 0.5
VAD_STOP_THRESHOLD = 0.3
VAD_QUIET_THRESHOLD = 0.05
VAD_STATE_SHAPE = (2, 1, 128)
VAD_CONTEXT_SIZE = 64

# Speech Segmentation
SAFETY_CHUNKS_BEFORE = 4

# End-of-Utterance Detection
EOU_MIN_SAMPLES = 4 * SAMPLE_RATE
EOU_OPTIMAL_SAMPLES = 8 * SAMPLE_RATE
EOU_CONFIDENCE_THRESHOLD = 0.9

# Processing Configuration
MAX_TRANSCRIPTION_QUEUE_SIZE = 256
MIN_SEGMENT_DURATION = 0.3

# LLM Streaming Configuration
LLM_MIN_TOKENS_FOR_TTS = 3
LLM_SENTENCE_DELIMITERS = ".!?"
LLM_MAX_TOKENS = 256

# TTS Configuration
TTS_SAMPLE_RATE = 24000

# Feature Flags
ENABLE_RECORDING = False
ENABLE_TRANSCRIPTION = True

# Model Paths
VAD_MODEL_PATH = "models/silero_vad.onnx"
EOU_MODEL_PATH = "models/smart_turn_v3.onnx"
WHISPER_MODEL = "mlx-community/whisper-small.en-mlx-q4"
LLM_MODEL = "mlx-community/LFM2-1.2B-4bit"
# LLM_MODEL = "mlx-community/Qwen3-0.6B-8bit"
TTS_MODEL = "hexgrad/Kokoro-82M"
TTS_VOICE = "af_heart"
TTS_SPEED = 1.0
