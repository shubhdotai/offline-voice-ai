from mlx_lm import load, generate
from typing import List, Dict


class LLMHandler:
    def __init__(
        self,
        model_name: str = "mlx-community/LFM2-1.2B-4bit",
        system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational."
    ):
        """Initialize LLM handler with model and tokenizer."""
        print(f"Loading LLM model: {model_name}")
        self.model, self.tokenizer = load(model_name)
        self.system_prompt = system_prompt
        print("LLM model loaded successfully")
    
    def generate_response(self, conversation_history: List[Dict[str, str]]) -> str:
        """
        Generate response from conversation history.
        
        Args:
            conversation_history: List of message dicts with 'role' and 'content'
            
        Returns:
            Generated response text
        """
        try:
            # Build messages with system prompt
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(conversation_history)
            
            # Apply chat template if available
            if self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True,
                    tokenize=False
                )
            else:
                # Fallback: simple prompt formatting
                prompt = self._format_messages_simple(messages)
            
            print(f"LLM Prompt (last 200 chars): ...{str(prompt)[-200:]}")
            
            # Generate response
            response = generate(
                self.model, 
                self.tokenizer, 
                prompt=prompt, 
                verbose=False,
                max_tokens=256
            )
            
            # mlx_lm.generate returns the full completion including the prompt
            # We need to extract only the new generated text
            if isinstance(prompt, str) and response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            print(f"LLM Response: {response}")
            return response.strip()
            
        except Exception as e:
            print(f"LLM generation error: {e}")
            return "I apologize, but I encountered an error processing your request."
    
    def _format_messages_simple(self, messages: List[Dict[str, str]]) -> str:
        """Simple fallback message formatting."""
        formatted = ""
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted += f"{role}: {content}\n"
        formatted += "Assistant: "
        return formatted