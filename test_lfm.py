from mlx_lm import load, generate

model, tokenizer = load("mlx-community/LFM2-1.2B-4bit")

prompt = "Hello"
if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )
response = generate(model, tokenizer, prompt=prompt, verbose=True)
print(f"Response: {response}")
