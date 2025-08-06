import os
import torch
import subprocess
from tqdm import tqdm
from transformers import AutoModelForTextToWaveform, AutoProcessor
from pydub import AudioSegment
import srt

# 📂 Input paths
srt_file = "sample_output_translated_ta.srt"   # Tamil SRT with speaker labels
output_dir = "tts_segments"
output_wav = "output.wav"

# 🧠 Load IndicParler-TTS model
print("📦 Loading ai4bharat/indic-parler-tts...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForTextToWaveform.from_pretrained("ai4bharat/indic-parler-tts").to(device)
processor = AutoProcessor.from_pretrained("ai4bharat/indic-parler-tts")

# 🧹 Cleanup and prepare output dir
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

    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        waveform = model(**inputs).waveform[0].cpu()

    segment_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
    processor.save_wav(waveform, segment_path, sampling_rate=16000)
    segment_paths.append(segment_path)

# 🔊 Stitch audio segments
print("🔗 Stitching segments into final audio...")
combined = AudioSegment.silent(duration=0)
for segment in segment_paths:
    audio = AudioSegment.from_wav(segment)
    combined += audio

combined.export(output_wav, format="wav")
print(f"✅ Output audio saved to {output_wav}")

# 🧽 Clean up intermediate files
for path in segment_paths:
    os.remove(path)
os.rmdir(output_dir)
