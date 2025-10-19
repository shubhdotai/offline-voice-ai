# llm_handler.py
"""LLM handler with true MLX streaming support"""
from typing import List, Dict, Iterator
from mlx_lm import load, generate, stream_generate
import threading
from config import LLM_MODEL, LLM_MAX_TOKENS, LLM_SENTENCE_DELIMITERS, LLM_MIN_TOKENS_FOR_TTS


class LLMHandler:
    """Handles LLM inference with streaming support"""
    
    def __init__(
        self,
        model_name: str = LLM_MODEL,
        system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational."
    ):
        print(f"Loading LLM: {model_name}")
        self.model, self.tokenizer = load(model_name)
        self.system_prompt = system_prompt
        # CRITICAL: Add lock to prevent concurrent access to MLX model
        self._generation_lock = threading.Lock()
        print("LLM loaded successfully")
    
    def stream_response(self, conversation_history: List[Dict[str, str]]) -> Iterator[str]:
        """
        Stream LLM response with true token-by-token generation from MLX.
        Yields complete sentences as they're formed.
        """
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(conversation_history)
            
            prompt = self._format_prompt(messages)
            print("Generating LLM response (streaming)...")
            
            # Buffer for accumulating tokens into sentences
            buffer = ""
            
            # Run generation under lock to avoid concurrent model access
            with self._generation_lock:
                for response in stream_generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=LLM_MAX_TOKENS
                ):
                    # response.text contains the newly generated token(s)
                    token_text = response.text
                    buffer += token_text
                    
                    # Check if we've hit a sentence delimiter
                    if any(delimiter in token_text for delimiter in LLM_SENTENCE_DELIMITERS):
                        # Extract complete sentences from buffer
                        sentences = self._extract_complete_sentences(buffer)
                        
                        for sentence in sentences['complete']:
                            if sentence:
                                print(f"[LLM] Sentence: {sentence}")
                                yield sentence
                        
                        # Keep incomplete part in buffer
                        buffer = sentences['remaining']
            
            # Yield any remaining text in buffer
            if buffer.strip():
                print(f"[LLM] Final: {buffer.strip()}")
                yield buffer.strip()
        
        except Exception as e:
            print(f"LLM streaming error: {e}")
            import traceback
            traceback.print_exc()
            yield "I apologize, but I encountered an error."
    
    def stream_response_batched(self, conversation_history: List[Dict[str, str]]) -> Iterator[str]:
        """
        Legacy batched response (generates all tokens first, then streams sentences).
        Kept for backwards compatibility or fallback.
        """
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(conversation_history)
            
            prompt = self._format_prompt(messages)
            print("Generating LLM response (batched)...")
            
            # Run generation under lock to avoid concurrent model access
            with self._generation_lock:
                full_text = generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=LLM_MAX_TOKENS
                )
            
            full_text = (full_text or "").strip()
            if not full_text:
                print("[LLM] Empty generation")
                yield "I'm sorry, I couldn't think of anything to say."
                return
            
            for sentence in self._split_into_sentences(full_text):
                if sentence:
                    print(f"[LLM] Sentence: {sentence}")
                    yield sentence
        
        except Exception as e:
            print(f"LLM streaming error: {e}")
            import traceback
            traceback.print_exc()
            yield "I apologize, but I encountered an error."
    
    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into a prompt string"""
        if self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=False
            )
        
        # Fallback formatting
        formatted = ""
        for msg in messages:
            formatted += f"{msg['role'].capitalize()}: {msg['content']}\n"
        formatted += "Assistant: "
        return formatted
    
    def _extract_complete_sentences(self, buffer: str) -> Dict[str, any]:
        """
        Extract complete sentences from buffer while preserving incomplete text.
        Returns dict with 'complete' (list of sentences) and 'remaining' (buffer).
        """
        complete_sentences = []
        remaining = buffer
        
        # Find all sentence delimiters
        last_delimiter_pos = -1
        for i, char in enumerate(buffer):
            if char in LLM_SENTENCE_DELIMITERS:
                last_delimiter_pos = i
        
        # If we found at least one delimiter, split there
        if last_delimiter_pos >= 0:
            complete_part = buffer[:last_delimiter_pos + 1]
            remaining = buffer[last_delimiter_pos + 1:]
            
            # Split complete part into sentences
            sentences = self._split_into_sentences(complete_part)
            
            # Apply minimum token threshold to avoid tiny chunks
            for sentence in sentences:
                if not complete_sentences:
                    complete_sentences.append(sentence)
                else:
                    word_count = len(sentence.split())
                    if word_count < LLM_MIN_TOKENS_FOR_TTS:
                        # Merge with previous sentence if too short
                        complete_sentences[-1] = (complete_sentences[-1] + " " + sentence).strip()
                    else:
                        complete_sentences.append(sentence)
        
        return {
            'complete': complete_sentences,
            'remaining': remaining
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split generated text into sensible sentence chunks"""
        raw_sentences: List[str] = []
        buffer = ""
        
        for char in text:
            buffer += char
            if char in LLM_SENTENCE_DELIMITERS:
                chunk = buffer.strip()
                if chunk:
                    raw_sentences.append(chunk)
                buffer = ""
        
        if buffer.strip():
            raw_sentences.append(buffer.strip())
        
        if not raw_sentences:
            return [text]
        
        merged: List[str] = []
        for sentence in raw_sentences:
            if not merged:
                merged.append(sentence)
                continue
            
            word_count = len(sentence.split())
            if word_count < LLM_MIN_TOKENS_FOR_TTS:
                merged[-1] = (merged[-1] + " " + sentence).strip()
            else:
                merged.append(sentence)
        
        return merged