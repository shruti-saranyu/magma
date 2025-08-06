import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
import srt

# ✅ Add local parler-tts path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "parler-tts")))

from parler_tts.modeling_parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.configuration_parler_tts import (
    ParlerTTSConfig,
    ParlerTTSConfigTextEncoder,
    ParlerTTSConfigAudioEncoder,
    ParlerTTSConfigDecoder,
)
from transformers import AutoProcessor

# 📂 File paths
srt_file = "sample_output_translated_ta.srt"
output_dir = "tts_segments"
output_wav = "output.wav"

# 🚀 Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")
print("📦 Loading model & config...")

# ✅ Manually define submodel configs
config = ParlerTTSConfig(
    text_encoder=ParlerTTSConfigTextEncoder(
        pretrained_model_name_or_path="ai4bharat/indic-parler-tts-text-encoder"
    ),
    audio_encoder=ParlerTTSConfigAudioEncoder(
        pretrained_model_name_or_path="ai4bharat/indic-parler-tts-audio-encoder"
    ),
    decoder=ParlerTTSConfigDecoder(
        pretrained_model_name_or_path="ai4bharat/indic-parler-tts-decoder"
    )
)

# ✅ Load model with full config
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ai4bharat/indic-parler-tts",
    config=config
).to(device)

# ✅ Load processor
processor = AutoProcessor.from_pretrained("ai4bharat/indic-parler-tts")
sampling_rate = model.config.sampling_rate

# 🔊 Prepare output folder
os.makedirs(output_dir, exist_ok=True)

# 📖 Parse SRT
with open(srt_file, "r", encoding="utf-8") as f:
    subtitles = list(srt.parse(f.read()))

# 🎤 Generate audio segments
segment_paths = []
print("🔊 Generating segments...")
for i, sub in enumerate(tqdm(subtitles)):
    text = sub.content.strip()
    if not text:
        continue

    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**inputs, do_sample=True)

    waveform = generated.cpu().numpy().squeeze()
    waveform = np.clip(waveform, -1, 1)

    segment_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
    waveform_int = (waveform * 32767).astype(np.int16)
    audio = AudioSegment(
        waveform_int.tobytes(),
        frame_rate=sampling_rate,
        sample_width=2,
        channels=1
    )
    audio.export(segment_path, format="wav")
    segment_paths.append(segment_path)

# 🧵 Stitch segments
print("🔗 Stitching...")
combined = AudioSegment.silent(duration=0)
for path in segment_paths:
    combined += AudioSegment.from_wav(path)

combined.export(output_wav, format="wav")
print(f"✅ Final audio saved to: {output_wav}")

# 🧽 Cleanup
for path in segment_paths:
    os.remove(path)
os.rmdir(output_dir)
