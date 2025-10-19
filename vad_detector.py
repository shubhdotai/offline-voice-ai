# vad_detector.py
"""Voice Activity Detection and End-of-Utterance detection"""
import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor
from config import (
    VAD_MODEL_PATH, EOU_MODEL_PATH, SAMPLE_RATE,
    VAD_ALPHA, VAD_STATE_SHAPE, VAD_CONTEXT_SIZE,
    EOU_MIN_SAMPLES, EOU_OPTIMAL_SAMPLES, EOU_CONFIDENCE_THRESHOLD
)


class VADDetector:
    """Voice Activity Detection using Silero VAD"""
    
    def __init__(self, model_path: str = VAD_MODEL_PATH):
        print(f"Loading VAD: {model_path}")
        self.session = ort.InferenceSession(model_path)
        self.state = np.zeros(VAD_STATE_SHAPE, dtype=np.float32)
        self.context = np.zeros((1, VAD_CONTEXT_SIZE), dtype=np.float32)
        self.smoothed_prob = 0.0
        print("VAD loaded")
    
    def process_chunk(self, chunk: np.ndarray) -> float:
        """Process audio chunk and return smoothed VAD probability"""
        # Prepare input with context
        audio_input = np.concatenate([self.context, chunk.reshape(1, -1)], axis=1)
        
        # Run VAD inference
        output, self.state = self.session.run(
            None,
            {
                'input': audio_input,
                'state': self.state,
                'sr': np.array([SAMPLE_RATE], dtype=np.int64)
            }
        )
        
        # Update context for next chunk
        self.context = audio_input[:, -VAD_CONTEXT_SIZE:]
        
        # Apply exponential smoothing
        raw_prob = float(output[0][0])
        self.smoothed_prob = VAD_ALPHA * raw_prob + (1.0 - VAD_ALPHA) * self.smoothed_prob
        
        return self.smoothed_prob
    
    def reset(self):
        """Reset VAD state"""
        self.state = np.zeros(VAD_STATE_SHAPE, dtype=np.float32)
        self.context = np.zeros((1, VAD_CONTEXT_SIZE), dtype=np.float32)
        self.smoothed_prob = 0.0


class EndOfUtteranceDetector:
    """Detect end of user utterance using ML model"""
    
    def __init__(self, model_path: str = EOU_MODEL_PATH):
        print(f"Loading EOU: {model_path}")
        
        self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)
        
        # Optimize ONNX session
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, sess_options=options)
        self.audio_buffer = np.array([], dtype=np.float32)
        print("EOU loaded")
    
    def add_audio(self, chunk: np.ndarray):
        """Add audio chunk to buffer"""
        self.audio_buffer = np.concatenate([self.audio_buffer, chunk])
        
        # Keep only recent audio (8 seconds max)
        if len(self.audio_buffer) > EOU_OPTIMAL_SAMPLES:
            self.audio_buffer = self.audio_buffer[-EOU_OPTIMAL_SAMPLES:]
    
    def has_enough_audio(self) -> bool:
        """Check if buffer has minimum required audio"""
        return len(self.audio_buffer) >= EOU_MIN_SAMPLES
    
    def detect(self) -> dict:
        """Detect if utterance has ended"""
        if not self.has_enough_audio():
            return {'ended': False, 'confidence': 0.0}
        
        try:
            # Use up to optimal amount of audio
            audio_length = min(len(self.audio_buffer), EOU_OPTIMAL_SAMPLES)
            audio = self.audio_buffer[-audio_length:]
            
            # Extract features
            inputs = self.feature_extractor(
                audio,
                sampling_rate=SAMPLE_RATE,
                return_tensors="np",
                padding="max_length",
                max_length=EOU_OPTIMAL_SAMPLES,
                truncation=True,
                do_normalize=True
            )
            
            # Run inference
            features = np.expand_dims(
                inputs.input_features.squeeze(0).astype(np.float32), 
                axis=0
            )
            outputs = self.session.run(None, {"input_features": features})
            confidence = float(outputs[0][0].item())
            
            return {
                'ended': confidence > EOU_CONFIDENCE_THRESHOLD,
                'confidence': confidence
            }
        
        except Exception as e:
            print(f"EOU detection error: {e}")
            return {'ended': False, 'confidence': 0.0}
    
    def reset(self):
        """Clear audio buffer"""
        self.audio_buffer = np.array([], dtype=np.float32)