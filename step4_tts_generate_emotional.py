import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from pydub import AudioSegment
import srt

# Add path to local `parler-tts`
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "parler-tts")))

from parler_tts.modeling_parler_tts import ParlerTTSForConditionalGeneration
from parler_tts.configuration_parler_tts import (
    ParlerTTSConfig,
    ParlerTTSConfigTextEncoder,
    ParlerTTSConfigAudioEncoder,
    ParlerTTSConfigDecoder
)
from transformers import AutoProcessor

# Paths
srt_file = "sample_output_translated_ta.srt"
output_dir = "tts_segments"
output_wav = "output.wav"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {device}")

# Load individual sub-model configs
print("📦 Loading configs...")
text_encoder_cfg = ParlerTTSConfigTextEncoder.from_pretrained("ai4bharat/indic-parler-tts-text-encoder")
audio_encoder_cfg = ParlerTTSConfigAudioEncoder.from_pretrained("ai4bharat/indic-parler-tts-audio-encoder")
decoder_cfg = ParlerTTSConfigDecoder.from_pretrained("ai4bharat/indic-parler-tts-decoder")

# Compose full config
config = ParlerTTSConfig(
    text_encoder=text_encoder_cfg,
    audio_encoder=audio_encoder_cfg,
    decoder=decoder_cfg
)

# Load model
print("🔁 Loading model...")
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "ai4bharat/indic-parler-tts",
    config=config
).to(device)

# Load processor
processor = AutoProcessor.from_pretrained("ai4bharat/indic-parler-tts")
sampling_rate = model.config.sampling_rate

# Create output dir
os.makedirs(output_dir, exist_ok=True)

# Read SRT
with open(srt_file, "r", encoding="utf-8") as f:
    subtitles = list(srt.parse(f.read()))

# Generate audio segments
print("🔊 Generating speech segments...")
segment_paths = []

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

# Stitch together
print("🔗 Stitching audio...")
combined = AudioSegment.silent(duration=0)
for path in segment_paths:
    combined += AudioSegment.from_wav(path)

combined.export(output_wav, format="wav")
print(f"✅ Output saved to {output_wav}")

# Cleanup
for path in segment_paths:
    os.remove(path)
os.rmdir(output_dir)
