import os
import torch
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
import srt

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# 📂 Input/output
srt_file = "sample_output_translated_ta.srt"
output_dir = "tts_segments"
output_wav = "output.wav"

# 🚀 Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# 🧠 Load model & processor from HF Hub
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "ai4bharat/indic-parler-tts", trust_remote_code=True
).to(device)

processor = AutoProcessor.from_pretrained(
    "ai4bharat/indic-parler-tts", trust_remote_code=True
)

sampling_rate = model.config.sampling_rate

# 📁 Output dir
os.makedirs(output_dir, exist_ok=True)

# 📖 Parse SRT
with open(srt_file, "r", encoding="utf-8") as f:
    subtitles = list(srt.parse(f.read()))

# 🔁 Generate audio segments
segment_paths = []
print("🔊 Generating speech...")

for i, sub in enumerate(tqdm(subtitles)):
    text = sub.content.strip()
    if not text:
        continue

    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(**inputs, do_sample=True)

    waveform = output.cpu().numpy().squeeze()
    waveform = np.clip(waveform, -1, 1)

    segment_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
    waveform_int = (waveform * 32767).astype(np.int16)
    audio = AudioSegment(
        waveform_int.tobytes(), frame_rate=sampling_rate, sample_width=2, channels=1
    )
    audio.export(segment_path, format="wav")
    segment_paths.append(segment_path)

# 🔗 Stitch audio
print("🔗 Stitching audio...")
combined = AudioSegment.silent(duration=0)
for path in segment_paths:
    combined += AudioSegment.from_wav(path)

combined.export(output_wav, format="wav")
print(f"✅ Output saved to {output_wav}")

# 🧽 Cleanup
for path in segment_paths:
    os.remove(path)
os.rmdir(output_dir)
