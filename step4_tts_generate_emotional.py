import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import ParlerTTSForConditionalGeneration, AutoProcessor
from pydub import AudioSegment
import srt

# 📂 Input paths
srt_file = "sample_output_translated_ta.srt"
output_dir = "tts_segments"
output_wav = "output.wav"

# 🚀 Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 🧠 Load model with updated architecture
print("📦 Loading ai4bharat/indic-parler-tts...")
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
processor = AutoProcessor.from_pretrained("ai4bharat/indic-parler-tts")

# 🧹 Prepare output directory
os.makedirs(output_dir, exist_ok=True)

# 📖 Read SRT file
with open(srt_file, "r", encoding="utf-8") as f:
    subtitles = list(srt.parse(f.read()))

# 🔁 TTS generation for each subtitle
segment_paths = []
print("🔊 Generating speech segments...")

for i, sub in enumerate(tqdm(subtitles)):
    text = sub.content.strip()
    if not text:
        continue

    # Generate audio with proper sampling
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**inputs, do_sample=True)
    
    # Convert to numpy array and normalize
    waveform = generated.cpu().numpy().squeeze()
    waveform = np.clip(waveform, -1, 1)  # Prevent clipping
    
    # Save as WAV using pydub
    segment_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
    waveform_int = (waveform * 32767).astype(np.int16)
    audio = AudioSegment(
        waveform_int.tobytes(),
        frame_rate=16000,
        sample_width=2,
        channels=1
    )
    audio.export(segment_path, format="wav")
    segment_paths.append(segment_path)

# 🔊 Stitch audio segments
print("🔗 Stitching segments into final audio...")
combined = AudioSegment.silent(duration=0)
for path in segment_paths:
    combined += AudioSegment.from_wav(path)

combined.export(output_wav, format="wav")
print(f"✅ Output saved to {output_wav}")

# 🧽 Cleanup
for path in segment_paths:
    os.remove(path)
os.rmdir(output_dir)