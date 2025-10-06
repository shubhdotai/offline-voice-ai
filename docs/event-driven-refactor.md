# Event-Driven Refactor Blueprint

This blueprint outlines how to refactor the current voice pipeline into an event-driven architecture. It highlights the core concepts, pseudo code sketches, and the concrete changes required across the project to support an event bus + handler pattern.

---

## 1. Event System Overview

### Goals
- Decouple detection, transcription, LLM, and TTS stages via asynchronous events.
- Allow any component to subscribe to or publish events without tight coupling.
- Support prioritised or cancellable events (e.g., interruptions) out of the box.

### Core Constructs
- **EventBus**: central async dispatcher managing topics/queues.
- **Events**: typed payloads with metadata (`type`, `payload`, `timestamp`, optional `correlation_id`).
- **Handlers**: async callables registered for specific event types. Optionally carry priorities or filters.
- **Middleware** *(optional)*: cross-cutting concerns like logging, metrics, fault-retry.

```pseudo
class Event:
    type: str
    payload: dict
    timestamp: datetime
    correlation_id: UUID | None

class EventBus:
    registry: Dict[str, List[Handler]]
    queue: asyncio.Queue[Event]

    async def publish(event: Event):
        await queue.put(event)

    def subscribe(event_type: str, handler: Handler, *, priority=0):
        registry[event_type].insert_by_priority(handler)

    async def start():
        while running:
            event = await queue.get()
            for handler in registry[event.type]:
                await handler(event)
```

---

## 2. Pipeline Decomposition

### Event Types (illustrative)
- `vad.speech_starting`
- `vad.speech_started`
- `vad.speech_stopping`
- `vad.speech_ended`
- `transcription.request`
- `transcription.completed`
- `llm.request`
- `llm.completed`
- `tts.request`
- `tts.completed`
- `playback.started`
- `playback.finished`
- `playback.interrupted`

### Component Responsibilities
| Component | Subscribes to | Publishes |
|-----------|---------------|-----------|
| **VAD/Detector** | (none) | `vad.*`, `transcription.request`
| **Transcription Worker** | `transcription.request` | `transcription.completed`
| **Conversation Orchestrator** | `transcription.completed`, `playback.finished`, `playback.interrupted` | `llm.request`
| **LLM Worker** | `llm.request` | `llm.completed`
| **TTS Worker** | `llm.completed` | `tts.completed`
| **Playback Manager** | `tts.completed`, UI notifications | `playback.started`, `playback.finished`, `playback.interrupted`
| **UI Gateway** (WebSocket) | `playback.*`, `transcription.completed` | `playback.interrupted` + user control events

---

## 3. Pseudo Code Flow

### 3.1 Detector Publishing
```pseudo
async def process_audio_chunk(chunk):
    state = vad.analyse(chunk)
    if state transitioned to STARTING:
        await bus.publish(Event("vad.speech_starting", payload={...}))
    if state transitioned to SPEAKING:
        await bus.publish(Event("vad.speech_started", payload={...}))
    if state transitioned to STOPPING and captured_segment:
        await bus.publish(Event(
            "transcription.request",
            payload={"audio": captured_segment, "correlation_id": segment_id}
        ))
    if state transitioned to QUIET:
        await bus.publish(Event("vad.speech_ended", payload={...}))
```

### 3.2 Transcription Worker
```pseudo
@bus.subscribe("transcription.request")
async def handle_transcription(event):
    transcript = await run_whisper(event.payload["audio"])
    await bus.publish(Event(
        "transcription.completed",
        payload={
            "text": transcript,
            "correlation_id": event.payload["correlation_id"]
        }
    ))
```

### 3.3 Conversation Orchestrator
```pseudo
@bus.subscribe("transcription.completed")
async def handle_user_utterance(event):
    conversation.append({"role": "user", "content": event.payload["text"]})
    await bus.publish(Event("llm.request", payload={"conversation": conversation}))

@bus.subscribe("playback.finished")
async def resume_listening(event):
    detector.enable_listening()

@bus.subscribe("playback.interrupted")
async def handle_interrupt(event):
    cancel_pending_tasks()
    detector.enable_listening()
```

### 3.4 LLM & TTS Workers
```pseudo
@bus.subscribe("llm.request")
async def handle_llm(event):
    response = await run_llm(event.payload["conversation"])
    await bus.publish(Event("llm.completed", payload={"text": response}))

@bus.subscribe("llm.completed")
async def handle_tts_request(event):
    await bus.publish(Event("tts.request", payload={"text": event.payload["text"]}))

@bus.subscribe("tts.request")
async def handle_tts(event):
    audio = await run_tts(event.payload["text"])
    await bus.publish(Event("tts.completed", payload={"audio": audio}))
```

### 3.5 Playback Manager
```pseudo
@bus.subscribe("tts.completed")
async def stream_audio(event):
    await notify_ui(audio=event.payload["audio"])
    await bus.publish(Event("playback.started", payload={...}))

@ui.on("audio_finished")
async def ui_finished():
    await bus.publish(Event("playback.finished", payload={...}))

@ui.on("audio_interrupted")
async def ui_interrupted():
    await bus.publish(Event("playback.interrupted", payload={...}))
```

---

## 4. Required Code Changes

### 4.1 Infrastructure
- `server.py`
  - Extract reusable `Event`, `EventBus`, and `EventHandler` classes into a new module (e.g., `event_bus.py`).
  - Replace direct method calls in `VoicePipelineOrchestrator` with event publications/subscriptions.
  - Convert `SpeechDetector` to publish events (`bus.publish`) rather than returning event lists.
  - Run the event bus loop as an application-level background task.
- `event_bus.py` *(new)*
  - Implement async publish/subscribe, handler registration, optional middleware, and graceful shutdown.

### 4.2 Components -> Event Handlers
- **Detector**
  - Inject `EventBus` reference; publish `vad.*` and `transcription.request` events.
  - Remove `PipelineEvent` enum in favor of string event names.
- **Transcription Worker**
  - Subscribe to `transcription.request`; rework queue logic to rely on bus.
  - When disabled, simply unregister handler or short-circuit on publish.
- **LLM/TTS**
  - Wrap current `process_llm_and_tts` pipeline into discrete handlers for `llm.request`, `llm.completed`, `tts.request`, `tts.completed`.
  - Maintain conversation state inside a dedicated orchestrator handler or attach to event payload via `correlation_id`.
- **Playback**
  - Replace direct websocket sends with event-driven notifications (e.g., `playback.started` -> UI message).
  - Listen for UI-originated events (`audio_finished`, `audio_interrupted`) and republish them on the bus.

### 4.3 UI Gateway Adjustments
- WebSocket layer listens for bus events instead of calling orchestrator methods directly.
- Map inbound UI commands to events (e.g., `start_listening` → `ui.control.start`).
- Optionally move UI gateway into its own module for clarity.

### 4.4 Concurrency & Interruption
- Use per-event `correlation_id` to tie together utterance lifecycle (speech detection → transcription → reply → playback).
- Implement cancellation by publishing an interrupt event that downstream handlers respect (e.g., TTS handler ignores requests for a cancelled `correlation_id`).

### 4.5 Testing / Observability
- Provide test harness for `EventBus` to assert handler invocation order and error handling.
- Add logging middleware to trace event flow.
- Update integration tests to assert behaviour via emitted events rather than method outputs.

---

## 5. Migration Strategy
1. **Introduce EventBus skeleton** running alongside existing orchestration.
2. **Incrementally migrate** components (transcription first, then LLM/TTS, then playback) to events.
3. **Remove legacy direct calls** once coverage and behaviour match.
4. **Document event contracts** (event names, payload schemas) for future contributors.

---

## 6. Optional Enhancements
- Add priority queues or topic partitions for high/low latency events.
- Support retries / dead-letter queues for failed handlers.
- Allow external integrations to subscribe via WebSockets or HTTP webhooks.
- Introduce metrics (counts, latencies) per event type for monitoring.

---

This plan should provide a clear roadmap for evolving the pipeline into a loosely coupled, event-driven system without diving into concrete Python implementation details.
