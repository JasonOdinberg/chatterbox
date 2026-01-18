import librosa
import numpy as np
import soundfile as sf
import tempfile
import torch
import gradio as gr
from chatterbox.vc import ChatterboxVC


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


model = ChatterboxVC.from_pretrained(DEVICE)
def _crossfade(a, b, overlap_len):
    if overlap_len <= 0:
        return np.concatenate([a, b])
    overlap_len = min(overlap_len, len(a), len(b))
    if overlap_len <= 0:
        return np.concatenate([a, b])
    fade_out = np.linspace(1.0, 0.0, overlap_len, dtype=a.dtype)
    fade_in = np.linspace(0.0, 1.0, overlap_len, dtype=b.dtype)
    blended = a[-overlap_len:] * fade_out + b[:overlap_len] * fade_in
    return np.concatenate([a[:-overlap_len], blended, b[overlap_len:]])


def _chunk_indices(num_samples, chunk_samples, overlap_samples):
    if chunk_samples <= 0 or num_samples <= chunk_samples:
        return [(0, num_samples)]
    step = max(chunk_samples - overlap_samples, 1)
    indices = []
    start = 0
    while start < num_samples:
        end = min(start + chunk_samples, num_samples)
        indices.append((start, end))
        if end == num_samples:
            break
        start += step
    return indices


def _generate_chunk(chunk_audio, sample_rate, target_voice_path):
    if hasattr(model, "generate_from_audio"):
        wav = model.generate_from_audio(chunk_audio, sample_rate, target_voice_path=target_voice_path)
        return wav.squeeze(0).numpy()

    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_wav:
        sf.write(temp_wav.name, chunk_audio, sample_rate)
        wav = model.generate(temp_wav.name, target_voice_path=target_voice_path)
        return wav.squeeze(0).numpy()


def generate(audio, target_voice_path, chunk_seconds, overlap_seconds):
    if target_voice_path:
        model.set_target_voice(target_voice_path)

    audio_samples, sample_rate = librosa.load(audio, sr=None, mono=True)
    if chunk_seconds <= 0:
        wav = _generate_chunk(audio_samples, sample_rate, target_voice_path=None)
        return model.sr, wav

    chunk_samples = int(chunk_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)
    chunks = _chunk_indices(len(audio_samples), chunk_samples, overlap_samples)

    generated = []
    for start, end in chunks:
        chunk_audio = audio_samples[start:end]
        wav = _generate_chunk(chunk_audio, sample_rate, target_voice_path=None)
        generated.append(wav)

    if not generated:
        return model.sr, np.array([], dtype=np.float32)

    overlap_out = int(overlap_seconds * model.sr)
    stitched = generated[0]
    for segment in generated[1:]:
        stitched = _crossfade(stitched, segment, overlap_out)

    return model.sr, stitched


demo = gr.Interface(
    generate,
    [
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input audio file"),
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Target voice audio file (if none, the default voice is used)", value=None),
        gr.Slider(minimum=0, maximum=120, value=30, step=1, label="Chunk length (seconds, 0 = no chunking)"),
        gr.Slider(minimum=0, maximum=5, value=0.5, step=0.1, label="Chunk overlap (seconds)"),
    ],
    "audio",
)

if __name__ == "__main__":
    demo.launch()
