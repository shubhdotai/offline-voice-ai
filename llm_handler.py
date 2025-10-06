from mlx_lm import load, generate
from typing import List, Dict


class LLMHandler:
    def __init__(
        self,
        model_name: str = "mlx-community/LFM2-1.2B-4bit",
        system_prompt: str = "You are a helpful voice assistant. Keep responses concise and conversational."
    ):
        print(f"Loading LLM model: {model_name}")
        self.model, self.tokenizer = load(model_name)
        self.system_prompt = system_prompt
        print("LLM model loaded successfully")
    
    def generate_response(self, conversation_history: List[Dict[str, str]]) -> str:
        try:
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(conversation_history)
            
            if self.tokenizer.chat_template is not None:
                prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            else:
                prompt = self._format_messages_simple(messages)
            
            print(f"LLM Prompt (last 200 chars): ...{str(prompt)[-200:]}")
            
            response = generate(self.model, self.tokenizer, prompt=prompt, verbose=False, max_tokens=256)
            
            # Extract only the new generated text
            if isinstance(prompt, str) and response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            print(f"LLM Response: {response}")
            return response.strip()
        except Exception as e:
            print(f"LLM generation error: {e}")
            return "I apologize, but I encountered an error processing your request."
    
    def _format_messages_simple(self, messages: List[Dict[str, str]]) -> str:
        formatted = ""
        for msg in messages:
            formatted += f"{msg['role'].capitalize()}: {msg['content']}\n"
        formatted += "Assistant: "
        return formatted