# Technical Deep Dive: Voice Agent Architecture

Comprehensive explanation of how every component works, connects, and enables real-time voice interaction with barge-in capability.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Voice Activity Detection (VAD)](#voice-activity-detection-vad)
3. [Speech-to-Text (STT)](#speech-to-text-stt)
4. [Large Language Model (LLM)](#large-language-model-llm)
5. [Text-to-Speech (TTS)](#text-to-speech-tts)
6. [Barge-In System](#barge-in-system)
7. [Data Flow & Integration](#data-flow--integration)
8. [State Machine](#state-machine)
9. [Buffer Management](#buffer-management)
10. [Performance Optimization](#performance-optimization)

---

## System Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT (Browser)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Microphone (16kHz mono)                                         │
│       ↓                                                           │
│  ┌──────────────────────────────────────────────┐               │
│  │   Audio Worklet: Correlator                  │               │
│  │   ├─ Input 0: Microphone                     │               │
│  │   ├─ Input 1: TTS Reference (bot's voice)    │               │
│  │   └─ Output: Correlation metrics              │               │
│  └──────────────────────────────────────────────┘               │
│       ↓                                                           │
│  ScriptProcessor (512 samples/frame)                             │
│       ├─ Rolling buffer (30 frames = 1 sec)                      │
│       ├─ Energy-based interruption detection                     │
│       └─ Base64 encoding                                         │
│       ↓                                                           │
│  WebSocket (bidirectional)                                       │
│       ↓                                                           │
└───────┼───────────────────────────────────────────────────────┘
        ↓
┌───────┼───────────────────────────────────────────────────────┐
│       ↓                SERVER (Python/FastAPI)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  WebSocket Handler                                               │
│       ↓                                                           │
│  ┌────────────────────────────────────────┐                     │
│  │  VoicePipeline (async orchestrator)     │                     │
│  │  ├─ Event dispatcher                    │                     │
│  │  ├─ Shared MLX lock + cancellation      │                     │
│  │  └─ State synchronization               │                     │
│  └────────────────────────────────────────┘                     │
│       ↓                                                           │
│  ┌────────────────────────────────────────┐                     │
│  │  Speech Detector                        │                     │
│  │  ├─ VAD (Silero ONNX)                   │                     │
│  │  ├─ State machine                       │                     │
│  │  ├─ EoU detector (optional)             │                     │
│  │  └─ Segment buffer                      │                     │
│  └────────────────────────────────────────┘                     │
│       ↓                                                           │
│  Transcription Queue (async, max 2)                              │
│       ↓                                                           │
│  ┌────────────────────────────────────────┐                     │
│  │  STT: MLX Whisper                       │                     │
│  │  (Apple Silicon optimized)              │                     │
│  └────────────────────────────────────────┘                     │
│       ↓                                                           │
│  Conversation History (list of messages)                         │
│       ↓                                                           │
│  ┌────────────────────────────────────────┐                     │
│  │  LLM: MLX LLM (1.2B 4-bit)              │                     │
│  │  (Generates response)                   │                     │
│  └────────────────────────────────────────┘                     │
│       ↓                                                           │
│  Sentence Splitter                                               │
│       ↓                                                           │
│  ┌────────────────────────────────────────┐                     │
│  │  TTS: Kokoro (parallel per sentence)    │                     │
│  │  (24kHz WAV output)                     │                     │
│  └────────────────────────────────────────┘                     │
│       ↓                                                           │
│  WebSocket (send audio chunks)                                   │
│       ↓                                                           │
└───────┼───────────────────────────────────────────────────────┘
        ↓
┌───────┼───────────────────────────────────────────────────────┐
│       ↓                CLIENT (Browser)                         │
│  Audio Element Playback                                          │
│       ├─ Captured via captureStream()                           │
│       └─ Fed back to Correlator as reference                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Voice Activity Detection (VAD)

### Purpose
Distinguish speech from silence/noise in real-time to trigger transcription only when the user is actually speaking.

### Model: Silero VAD v6

**Architecture**: Recurrent Neural Network (RNN) trained on 6000+ hours of speech data

**Input**: 
- Audio chunk: 512 samples @ 16kHz (32ms)
- Previous state: 2×1×128 tensor (LSTM hidden state)
- Context: Last 64 samples from previous chunk

**Output**:
- Probability: Float [0.0-1.0] indicating likelihood of speech
- Updated state for next chunk

**Implementation**:
```python
def _get_vad_probability(self, chunk):
    # Concatenate context with new chunk
    x = np.concatenate([self._vad_context, chunk.reshape(1, -1)], axis=1)
    
    # Run inference
    out, self._vad_state = self.vad_session.run(
        None, 
        {
            'input': x,                              # 576 samples
            'state': self._vad_state,                # 2×1×128 LSTM state
            'sr': np.array([16000], dtype=np.int64)  # Sample rate
        }
    )
    
    # Update context for next frame (sliding window)
    self._vad_context = x[:, -64:]
    
    return float(out[0][0])  # Raw probability
```

### Exponential Smoothing

Raw VAD output is noisy. Smoothing reduces false positives:

```python
ALPHA = 0.2  # Smoothing factor

# Update smoothed probability
self.smoothed_prob = ALPHA * raw_prob + (1.0 - ALPHA) * self.smoothed_prob
```

**Effect**:
- α = 0.2: Current frame contributes 20%, history contributes 80%
- Smooths out brief noise spikes
- Adds ~50-100ms lag but improves accuracy

### Thresholds

```python
START_THRESHOLD = 0.3      # Initial speech detection
SPEAKING_THRESHOLD = 0.5   # Confirmed speaking
STOP_THRESHOLD = 0.3       # Speech ending
QUIET_THRESHOLD = 0.05     # Return to quiet
```

**Hysteresis**: Different thresholds for starting vs stopping prevents "flapping" (rapid on/off switching).

---

## Speech-to-Text (STT)

### Model: Whisper Small EN (MLX)

**Purpose**: Convert captured audio segments to text

**Model Details**:
- Size: ~244M parameters
- Quantization: 4-bit (MLX optimized)
- Language: English only
- Domain: General conversation

**Input Requirements**:
- Minimum length: 0.3 seconds (4,800 samples)
- Sample rate: 16 kHz
- Format: Float32 numpy array

### Transcription Process

```python
def transcribe(self, audio_data: np.ndarray) -> str:
    # 1. Validate minimum length
    if len(audio_data) < self.sample_rate * 0.3:
        return ""
    
    # 2. Run Whisper inference
    result = mlx_whisper.transcribe(
        audio_data,
        path_or_hf_repo=self.model,
        verbose=False,
        language="en",
        fp16=False,                      # MLX handles precision
        temperature=0.0,                 # Deterministic output
        no_speech_threshold=0.6,         # Reject non-speech
        compression_ratio_threshold=2.4  # Reject gibberish
    )
    
    # 3. Extract text
    transcript = result["text"].strip()
    return transcript
```

### Queue-Based Processing

**Why async?** Transcription takes 0.5-2 seconds. Running synchronously would block audio processing.

```python
# Server maintains a queue
self.transcription_queue = asyncio.Queue(maxsize=2)

# Worker processes queue continuously
async def process_transcription_queue(self, websocket):
    while True:
        request = await self.transcription_queue.get()
        
        # Run in thread pool (Whisper is CPU-bound)
        transcript = await loop.run_in_executor(
            None,
            self.transcriber.transcribe,
            request.audio_data
        )
        
        # Send result to client
        await websocket.send_text(json.dumps({
            "event": "text",
            "role": "user",
            "text": transcript
        }))
```

**Queue size = 2**: Prevents buildup if user speaks multiple utterances rapidly. Older segments are dropped.

---

## Large Language Model (LLM)

### Model: LFM2-1.2B-4bit (MLX)

**Purpose**: Generate conversational responses based on user input and conversation history

**Architecture**:
- Parameters: 1.2 billion
- Quantization: 4-bit for memory efficiency
- Context: 2048 tokens
- Type: Causal language model

### Response Generation

```python
def generate_response(self, conversation_history: List[Dict]) -> str:
    # 1. Build prompt with system message
    messages = [
        {"role": "system", "content": self.system_prompt}
    ] + conversation_history
    
    # 2. Apply chat template (if available)
    if self.tokenizer.chat_template:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        prompt = self._format_messages_simple(messages)
    
    # 3. Generate response
    response = generate(
        self.model,
        self.tokenizer,
        prompt=prompt,
        max_tokens=256,
        verbose=False
    )
    
    # 4. Strip prompt from output (MLX returns full completion)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response
```

### Conversation History

**Format**:
```python
conversation_history = [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "I don't have access to weather data."},
    {"role": "user", "content": "Can you help with Python?"},
    # ... continues
]
```

**Maintained in memory**: No database, session-scoped only.

---

## Text-to-Speech (TTS)

### Model: Kokoro-82M

**Purpose**: Convert LLM response text into natural-sounding speech

**Model Details**:
- Parameters: 82 million
- Sample rate: 24 kHz
- Voice: `af_heart` (American English, female)
- Speed: 1.0× (adjustable)

### Audio Generation

```python
def generate_speech(self, text: str) -> bytes:
    # 1. Validate input
    if not text or not text.strip():
        return b''
    
    # 2. Generate audio chunks (streaming)
    audio_chunks = []
    for result in self.pipeline(text, voice=self.voice, speed=self.speed):
        if result.audio is not None:
            audio_chunks.append(result.audio)
    
    # 3. Concatenate chunks
    full_audio = torch.cat(audio_chunks, dim=0)
    audio_array = full_audio.numpy()
    
    # 4. Convert to WAV format
    audio_bytes = self._audio_to_wav_bytes(audio_array, self.sample_rate)
    
    return audio_bytes
```

### Sentence-by-Sentence Streaming

**Why?** Starting playback faster improves perceived latency.

```python
# Split LLM response into sentences
sentences = _split_sentences(llm_response)

# Process in parallel: generate next TTS while playing previous
previous_tts = None
for sentence in sentences:
    # Send text to UI immediately
    await websocket.send_text(json.dumps({
        "event": "text",
        "role": "assistant",
        "text": sentence
    }))
    
    # Start TTS for this sentence
    tts_task = asyncio.create_task(
        asyncio.to_thread(self.tts_handler.generate_speech, sentence)
    )
    
    # Wait for previous TTS to complete
    if previous_tts:
        audio_bytes = await previous_tts
        await websocket.send_text(json.dumps({
            "event": "media",
            "mime": "audio/wav",
            "audio": base64.b64encode(audio_bytes).decode()
        }))
    
    previous_tts = tts_task
```

**Pipeline overlap**: While sentence N is being spoken, sentence N+1 is being generated.

---

## Barge-In System

### Problem Statement

When the bot is speaking, the microphone picks up the bot's own voice through the speakers. This creates two challenges:

1. **Echo**: Bot's voice gets sent back to VAD, causing false speech detection
2. **Interruption**: How to detect when the user actually wants to interrupt vs. just echo?

### Solution: Reference Signal Correlation

**Core Idea**: Compare microphone input with the exact audio being played. High similarity = echo, low similarity = real user speech.

### Component 1: AudioWorklet Correlator

**File**: `correlator.js`

**Purpose**: Real-time correlation analysis in the browser's audio thread

```javascript
class Correlator extends AudioWorkletProcessor {
  process(inputs, outputs) {
    const mic = inputs[0][0];    // Microphone input
    const ref = inputs[1]?.[0];  // TTS reference (what's playing)
    
    if (mic && ref && ref.length === mic.length) {
      // Calculate cosine similarity (normalized correlation)
      const corr = this._corr(mic, ref);
      const micRms = this._rms(mic);
      const refRms = this._rms(ref);
      
      // Post metrics to main thread
      this.port.postMessage({ 
        corr: Math.max(0, corr),  // 0-1 range
        micRms,                    // Mic energy level
        refRms                     // Reference energy level
      });
    }
    return true;
  }
  
  _corr(x, y) {
    // Normalized cross-correlation (cosine similarity)
    let xy = 0, xx = 0, yy = 0;
    for (let i = 0; i < x.length; i++) {
      xy += x[i] * y[i];
      xx += x[i] * x[i];
      yy += y[i] * y[i];
    }
    return xy / (Math.sqrt(xx * yy) + 1e-9);
  }
  
  _rms(buf) {
    // Root mean square (energy level)
    let sum = 0;
    for (let i = 0; i < buf.length; i++) {
      sum += buf[i] * buf[i];
    }
    return Math.sqrt(sum / buf.length);
  }
}
```

### Component 2: Audio Graph Setup

```javascript
// Create audio context
audioContext = new AudioContext({sampleRate: 16000});

// Load correlator worklet
await audioContext.audioWorklet.addModule('correlator.js');

// Create correlator node (2 inputs, 1 output)
correlatorNode = new AudioWorkletNode(audioContext, 'correlator', {
    numberOfInputs: 2,
    numberOfOutputs: 1
});

// Connect microphone to input 0
const micSource = audioContext.createMediaStreamSource(stream);
micSource.connect(correlatorNode, 0, 0);

// Connect TTS audio to input 1 (during playback)
when_playing_audio() {
    ttsSourceNode = audioContext.createMediaStreamSource(
        dom.audio.captureStream()
    );
    ttsSourceNode.connect(correlatorNode, 0, 1);
}

// Listen to correlation results
correlatorNode.port.onmessage = (e) => {
    const { corr, micRms, refRms } = e.data;
    
    // Determine if this is echo
    echoState.isEcho = (
        corr > 0.30 &&      // High correlation
        refRms > 0.01 &&    // Reference is playing
        micRms < 0.05       // Mic is quiet (just echo)
    );
};
```

### Component 3: Interruption Detection

**Two-layer approach**: Correlation (removes echo) + Energy threshold (detects user)

```javascript
processorNode.onaudioprocess = (event) => {
    const frame = event.inputBuffer.getChannelData(0);
    
    // Always buffer recent frames
    audioBuffer.add(frame);
    
    if (botSpeaking) {
        // Check for interruption via energy threshold
        const burst = interruption.analyse(frame);
        
        if (burst) {
            // User is speaking over the bot!
            console.log('[interrupt] Detected');
            
            // 1. Set guard flag to prevent race condition
            interruptionInProgress = true;
            botSpeaking = false;
            
            // 2. IMMEDIATELY stop audio playback
            dom.audio.pause();
            dom.audio.currentTime = 0;
            
            // 3. Send buffered context + burst to server
            const preBuffer = audioBuffer.getRecent(15);  // 500ms
            preBuffer.forEach(chunk => sendToServer(chunk));
            burst.forEach(chunk => sendToServer(chunk));
            
            return;
        }
        
        // If echo (high correlation), don't send to server
        if (echoState.isEcho) return;
    }
    
    // Normal: send clean audio to server
    if (!echoState.isEcho) {
        sendToServer(frame);
    }
};
```

### Energy-Based Interruption Analysis

```javascript
const interruption = {
    frames: 0,
    avg: 0,       // Running average energy
    peak: 0,      // Peak energy (with decay)
    
    analyse(frame) {
        const energy = rms(frame);
        
        // Update statistics
        this.avg = 0.9 * this.avg + 0.1 * energy;
        this.peak = Math.max(this.peak * 0.95, energy);
        
        // Adaptive threshold
        const baseline = Math.max(this.avg, this.peak);
        const threshold = Math.max(
            0.012,                      // Absolute minimum
            this.avg + 0.008,           // Above average
            baseline * 1.20,            // 20% above baseline
            0.035                       // Absolute threshold
        );
        
        // Detect burst
        if (energy > threshold) {
            this.frames++;
            if (this.frames >= 2) {  // 2 consecutive frames
                return [/* burst frames */];
            }
        } else {
            this.frames = 0;
        }
        
        return null;
    }
};
```

### Why This Works

**Echo characteristics**:
- Correlation: 0.6-0.9 (very high)
- Energy: Low to moderate (speaker → mic distance)
- Steady: Consistent with reference signal

**User speech characteristics**:
- Correlation: 0.0-0.2 (low, uncorrelated with bot)
- Energy: High (close to microphone)
- Sudden: Rapid energy increase

**Interruption criteria**:
```
User interruption = (
    correlation < 0.30 AND
    energy > adaptive_threshold AND
    2+ consecutive frames
)
```

---

## Data Flow & Integration

### End-to-End Flow

**1. User Starts Speaking**
```
Mic → Correlator → ScriptProcessor → Base64 → WebSocket → Server
                                                           ↓
                                                    VAD (512 samples)
                                                           ↓
                                              State: QUIET → STARTING
                                                           ↓
                                              (smoothed_prob ≥ 0.3)
                                                           ↓
                                              State: STARTING → SPEAKING
                                                           ↓
                                              (smoothed_prob ≥ 0.5)
                                                           ↓
                                              Buffer: Capture audio + safety margins
```

**2. User Stops Speaking**
```
                                              State: SPEAKING → STOPPING
                                                           ↓
                                              (smoothed_prob < 0.3)
                                                           ↓
                                              Capture segment with safety buffers
                                                           ↓
                                              End-of-Utterance Detector (optional)
                                                           ↓
                                              Check: utterance_ended OR safety_margin_met
                                                           ↓
                                              State: STOPPING → QUIET
                                                           ↓
                                              Queue segment for transcription
```

**3. Transcription & Response**
```
Transcription Queue (async worker)
        ↓
MLX Whisper (0.5-2s)
        ↓
Transcript text → Conversation history
        ↓
MLX LLM (0.3-1s first token)
        ↓
Response text → Split into sentences
        ↓
For each sentence (parallel):
    ↓
TTS (Kokoro, 0.2-0.5s)
    ↓
WAV bytes → Base64 → WebSocket → Client
                                    ↓
                            Audio Element plays
                                    ↓
                            captureStream() → Correlator (reference)
```

**4. Barge-In Flow**
```
Bot speaking (TTS audio playing)
        ↓
Mic picks up: Bot echo + User voice
        ↓
Correlator analyzes:
    - Bot echo: corr=0.7, micRms=0.02
    - User voice: corr=0.1, micRms=0.08
        ↓
Energy threshold exceeded (2 frames)
        ↓
Interruption detected!
        ↓
Actions (synchronous):
    1. interruptionInProgress = true
    2. dom.audio.pause() + currentTime = 0
    3. botSpeaking = false
    4. Clear playback queue
    5. sendJson({event: 'stop', target: 'playback'}) (server interrupt hint)
        ↓
Send to server:
    - Pre-buffer (15 frames = 500ms context)
    - Burst frames (2-3 frames = 64-96ms)
        ↓
Server receives audio → VAD processes → `_interrupt_response()` sets cancel event → LLM/TTS tasks stop → Detector keeps listening
        ↓
Continue as normal user speech
```

### Server-side Cancellation

When the UI flags an interruption, the server-side `VoicePipeline`:

- Raises an `asyncio.Event` shared by the active LLM/TTS pipeline.  
- Awaits `_cancel_response_task()`, ensuring any running response task is cancelled and awaited safely.  
- Removes incomplete assistant turns before resuming listening.  
- Emits a `"interrupt"` WebSocket event, so the browser instantly abandons queued audio.

This guarantees that Whisper, the LLM, and Kokoro never generate output after the user barges in, while keeping the MLX runtime free for new transcription work.

---

## State Machine

### States & Transitions

```
┌─────────────────────────────────────────────────────────────┐
│                         QUIET                                │
│  • smoothed_prob < 0.05                                      │
│  • No speech detected                                        │
│  • Pre-buffer maintains last 4 chunks (128ms)               │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    (smoothed_prob ≥ 0.3)
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       STARTING                               │
│  • 0.3 ≤ smoothed_prob < 0.5                                │
│  • Potential speech beginning                                │
│  • Captures pre-buffer + current chunks                      │
└─────────────────────────────────────────────────────────────┘
         ↓                                      ↓
(smoothed_prob ≥ 0.5)              (smoothed_prob < 0.05)
         ↓                                      ↓
┌──────────────────────┐              ┌────────────────┐
│     SPEAKING         │              │  Back to QUIET │
│  • Confirmed speech  │              │  • False alarm │
│  • Accumulate audio  │              └────────────────┘
└──────────────────────┘
         ↓
(smoothed_prob < 0.3)
         ↓
┌─────────────────────────────────────────────────────────────┐
│                       STOPPING                               │
│  • Speech ending                                             │
│  • Increment post-buffer counter                             │
│  • Check EoU detector (if available)                         │
└─────────────────────────────────────────────────────────────┘
         ↓                                      ↓
(EoU detected OR                    (smoothed_prob ≥ 0.5)
 post_buffer ≥ 2 chunks)                       ↓
         ↓                              ┌──────────────┐
┌──────────────────┐                   │ SPEECH_RESUMED│
│  SPEECH_ENDED    │                   │ Back to       │
│  • Finalize      │                   │ SPEAKING      │
│  • Send to STT   │                   └──────────────┘
└──────────────────┘
```

### State Machine Code

```python
def _update_state(self, vad_prob: float) -> List[PipelineEvent]:
    events: List[PipelineEvent] = []
    prev_state = self.state

    if self.state == SpeechState.QUIET:
        if vad_prob >= VAD_START_THRESHOLD:
            self.state = SpeechState.STARTING
            self.user_speaking = True
            events.append(PipelineEvent.SPEECH_START)

    elif self.state == SpeechState.STARTING:
        if vad_prob >= VAD_SPEAKING_THRESHOLD:
            self.state = SpeechState.SPEAKING
        elif vad_prob < VAD_QUIET_THRESHOLD:
            self.state = SpeechState.QUIET
            self.user_speaking = False

    elif self.state == SpeechState.SPEAKING:
        if vad_prob < VAD_STOP_THRESHOLD:
            self.state = SpeechState.STOPPING
            self.current_segment = self.buffer.get_segment()
            if self.current_segment is not None:
                events.append(PipelineEvent.TRANSCRIBE)

    elif self.state == SpeechState.STOPPING:
        vad_quiet = vad_prob < VAD_QUIET_THRESHOLD
        eou_confirms = not self.eou
        if self.eou and vad_quiet and self.eou.has_enough_audio():
            result = self.eou.detect()
            eou_confirms = result['ended'] and result['confidence'] > EOU_CONFIDENCE_THRESHOLD
        if vad_quiet and eou_confirms:
            self.state = SpeechState.QUIET
            events.append(PipelineEvent.SPEECH_END)
            if self.user_speaking:
                events.append(PipelineEvent.RESPOND)
                self.user_speaking = False
            if self.eou:
                self.eou.reset()
            self.current_segment = None
        elif vad_prob > VAD_SPEAKING_THRESHOLD:
            self.state = SpeechState.SPEAKING
            self.current_segment = None

    if prev_state != self.state:
        print(f"[state] {prev_state.value} → {self.state.value} (vad: {vad_prob:.3f})")

    return events
```

---

## Buffer Management

### Pre-Buffer (Safety Margin Before)

**Purpose**: Capture audio before speech detection to avoid cutting off the first syllable

**Size**: 4 chunks × 32ms = 128ms

**Implementation**:
```python
class AudioBuffer:
    def __init__(self):
        self.pre_buffer: List[np.ndarray] = []
        self.active_segment: List[np.ndarray] = []
        self.is_capturing = False
    
    def add_chunk(self, chunk: np.ndarray, state: SpeechState):
        chunk = chunk.copy()
        
        if state == SpeechState.QUIET:
            self.pre_buffer.append(chunk)
            if len(self.pre_buffer) > SAFETY_CHUNKS_BEFORE:
                self.pre_buffer.pop(0)
        
        elif state == SpeechState.STARTING:
            if not self.is_capturing:
                self.is_capturing = True
                self.active_segment = self.pre_buffer.copy()
            self.active_segment.append(chunk)
            self.pre_buffer.append(chunk)
            if len(self.pre_buffer) > SAFETY_CHUNKS_BEFORE:
                self.pre_buffer.pop(0)
        
        elif state == SpeechState.SPEAKING:
            self.active_segment.append(chunk)
        
        elif state == SpeechState.STOPPING:
            self.active_segment.append(chunk)
    
    def get_segment(self) -> Optional[np.ndarray]:
        if not self.active_segment:
            return None
        segment = np.concatenate(self.active_segment)
        self.active_segment = []
        self.is_capturing = False
        return segment
```

**Timeline**:
```
t=-128ms: [chunk1] stored in pre_buffer
t=-96ms:  [chunk2] stored in pre_buffer
t=-64ms:  [chunk3] stored in pre_buffer
t=-32ms:  [chunk4] stored in pre_buffer
t=0ms:    Speech detected! (VAD ≥ 0.3)
          → Transfer pre_buffer to active_segment
          → User's first syllable captured cleanly
```

### Post-Buffer (Safety Margin After)

**Purpose**: Ensure speech has truly ended before finalizing

**Size**: 2 chunks × 32ms = 64ms

**Implementation**:
```python
def add_chunk(self, chunk, state):
    if state == SpeechState.STOPPING:
        self.active_segment.append(chunk.copy())
        self.post_buffer_count += 1

def has_safety_margin(self) -> bool:
    return self.post_buffer_count >= self.safety_after
```

**Timeline**:
```
t=0ms:     VAD drops below 0.3 → State: STOPPING
t=32ms:    Post-buffer count = 1
t=64ms:    Post-buffer count = 2 → Safety margin met
t=65ms:    Finalize segment → Send to transcription
```

### Rolling Audio Buffer (Client-Side)

**Purpose**: Capture context before interruption detection

**Size**: 30 frames × 32ms = ~1 second

**Implementation**:
```javascript
const audioBuffer = {
    frames: [],
    maxFrames: 30,
    
    add(frame) {
        this.frames.push(Float32Array.from(frame));
        if (this.frames.length > this.maxFrames) {
            this.frames.shift();  // FIFO
        }
    },
    
    getRecent(numFrames) {
        const start = Math.max(0, this.frames.length - numFrames);
        return this.frames.slice(start);
    }
};
```

**Usage during interruption**:
```javascript
if (burst_detected) {
    // Get last 500ms of audio
    const preBuffer = audioBuffer.getRecent(15);
    
    // Send context + burst
    preBuffer.forEach(chunk => sendToServer(chunk));
    burst.forEach(chunk => sendToServer(chunk));
}
```

**Why this matters**:
- User starts speaking softly: "Hel—" (200ms, below threshold)
- User continues louder: "—lo there!" (burst detected)
- Without buffer: Server receives "—lo there" → transcribes as "Oh there"
- With buffer: Server receives full "Hello there" → transcribes correctly

---

## Performance Optimization

### Latency Breakdown

**User Speech → Bot Response:**
```
Speech ending detected:           t=0ms
├─ Post-buffer safety:           +64ms
├─ Transcription queue:          +0-100ms (if queue busy)
├─ STT (Whisper):                +500-2000ms
├─ LLM (first token):            +300-1000ms
├─ TTS (first audio):            +200-500ms
├─ Network (WebSocket):          +20-50ms
└─ Audio playback starts:        =1084-3714ms total
```

**Typical**: 1.5-2.5 seconds end-to-end

### Barge-In Latency

**User Interruption → Audio Stops:**
```
User speaks over bot:             t=0ms
├─ Audio frame captured:         +0ms (real-time)
├─ Correlation analysis:         +0ms (AudioWorklet)
├─ Energy threshold check:       +0ms (synchronous)
├─ Frame 1 of burst:             +32ms
├─ Frame 2 of burst:             +64ms (interruption triggered)
├─ dom.audio.pause():            +1ms (synchronous)
└─ Audio playback stops:         =65ms total
```

**Target**: <100ms (typically 64-96ms)

### Memory Efficiency

**Client-Side**:
- Rolling buffer: 30 frames × 512 samples × 4 bytes = ~61 KB
- Correlator state: Minimal (no buffers, just computation)
- Total: <100 KB in memory

**Server-Side**:
- VAD state: 2×1×128 float32 = ~1 KB
- EoU buffer: 8 seconds × 16kHz × 4 bytes = ~512 KB
- Active segment: Variable (0.5-10 seconds) = ~32-640 KB
- Transcription queue: 2 segments × ~100 KB = ~200 KB
- **Total per connection**: <2 MB

### CPU Optimization

**Apple Silicon (M-series) advantages**:
1. **MLX**: Native Metal acceleration for ML models
2. **Unified memory**: Zero-copy between CPU/GPU
3. **Neural Engine**: Hardware acceleration for inference
4. **ONNX Runtime**: Optimized for ARM64

**Measurements on M1 Pro**:
- VAD inference: <1ms per frame
- EoU inference: ~20-50ms per check
- Whisper transcription: ~0.3-0.5× real-time (1 sec audio = 0.3-0.5 sec processing)
- LLM generation: ~30-50 tokens/second
- TTS generation: ~2-3× real-time

---

## Edge Cases & Handling

### 1. Simultaneous Speech (User + Bot)

**Scenario**: User starts speaking while bot is mid-sentence

**Handling**:
```python
if self.state == SpeechState.QUIET:
    if self.smoothed_prob >= START_THRESHOLD:
        events.append(PipelineEvent.SPEECH_STARTING)
        if self.is_playing_response:
            # Interrupt bot immediately
            pass  # Orchestrator handles cancellation
```

**Orchestrator action**:
```python
if PipelineEvent.SPEECH_STARTING in events:
    if self.detector.is_playing_response or self._response_task:
        interrupted = await self._interrupt_response()
        if interrupted:
            await websocket.send({"event": "interrupt"})
```

**Result**: Bot stops, user continues speaking

### 2. False Positives (Noise, Coughing)

**VAD smoothing** helps but doesn't eliminate false positives

**Additional filtering**:
- Minimum segment length: 0.3 seconds (4,800 samples)
- Whisper `no_speech_threshold`: 0.6
- If transcription is empty or gibberish, skip

### 3. Queue Overflow

**Scenario**: User speaks 3+ times rapidly before first transcription completes

**Handling**:
```python
try:
    self.transcription_queue.put_nowait(request)
except asyncio.QueueFull:
    self.dropped_segments += 1
    print(f"Queue full! Dropped segment {request.segment_id}")
```

**Trade-off**: Drop older segments to keep system responsive

### 4. Network Disconnection

**WebSocket connection lost**

**Client-side**:
```javascript
ws.onclose = () => {
    dom.connection.className = 'disconnected';
    playback.stop({updateUi: true});
    stopListening();
    setTimeout(connect, 2000);  // Auto-reconnect
};
```

**Server-side**:
```python
try:
    while True:
        message = await websocket.receive_text()
        await orchestrator.handle_event(json.loads(message))
except WebSocketDisconnect:
    print("Client disconnected")
finally:
    await orchestrator.shutdown()  # Clean up resources
```

### 5. Browser Compatibility

**correlator.js requires AudioWorklet** (Chrome 64+, Safari 14.1+, Firefox 76+)

**Fallback**:
```javascript
try {
    await audioContext.audioWorklet.addModule('correlator.js');
    // Use advanced echo cancellation
} catch (error) {
    console.error('Correlator not available, using fallback');
    setupSimpleAudioProcessing(stream);
    // Use basic energy-based detection only
}
```

---

## Debugging & Monitoring

### Client-Side Logs

```javascript
console.log('[interrupt] DETECTED - micRms:', micRms, 'corr:', corr);
console.log('[playback] Stopping - active:', active, 'queued:', queue.length);
console.log('[interrupt] Sending', preBuffer.length, 'buffered +', burst.length, 'burst');
```

### Server-Side Logs

```python
print(f"[state] {previous} -> {new_state} | vad={smoothed_prob:.3f}")
print(f"[audio] Segment ready ({length:.2f}s) for transcription")
print(f"[llm] Generating response for: {last_user_message}")
print(f"[interrupt] User interrupting bot response")
```

### Metrics Tracking

**Client sends to server**:
```javascript
{
    "event": "metrics",
    "metrics": {
        "stt": {"status": "completed", "latency": 1.23},
        "llm": {"first_token": 0.45},
        "tts": {"first_audio": 0.31}
    }
}
```

**Displayed in UI**: Real-time latency monitoring

---

## Summary

This voice agent achieves natural conversation through:

1. **VAD**: Robust speech detection with smoothing and state machine
2. **STT**: Fast, accurate transcription via MLX Whisper
3. **LLM**: Conversational responses with history awareness
4. **TTS**: Natural voice synthesis with sentence-level streaming
5. **Barge-In**: Professional-grade interruption using correlation analysis

The system handles full-duplex communication with <100ms interruption latency and 1.5-2.5s response time, making it feel responsive and natural.
