from kokoro import KPipeline
import torch
import soundfile as sf

# Initialize pipeline
pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')  # 'a' for American English
text = """Hello, My name is Kokoro. I am a text-to-speech model developed by Hexgrad AI."""

audio_chunks = []
for result in pipeline(text, voice='af_heart', speed=1):
    if result.audio is not None:
        audio_chunks.append(result.audio)

full_audio = torch.cat(audio_chunks, dim=0)
sf.write('audio.wav', full_audio.numpy(), 24000)