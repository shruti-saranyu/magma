import os
import re
import torch
import soundfile as sf
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForTextToWaveform

# 📦 Load Tamil-only TTS model (no emotion support)
print("📦 Loading ai4bharat/indic-tts-ta...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForTextToWaveform.from_pretrained("ai4bharat/indic-tts-ta").to(device)
processor = AutoProcessor.from_pretrained("ai4bharat/indic-tts-ta")

SPEAKER_PATTERN = re.compile(r"^Speaker\s+(\w+):\s*(.+)$")
os.makedirs("tts_segments", exist_ok=True)

def parse_srt(srt_path):
    with open(srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = []
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():
            start_end = lines[i+1].strip()
            text_line = lines[i+2].strip()
            match = SPEAKER_PATTERN.match(text_line)

            speaker = match.group(1) if match else "Unknown"
            text = match.group(2) if match else text_line

            entries.append({
                "index": lines[i].strip(),
                "start_end": start_end,
                "speaker": speaker,
                "text": text
            })
            i += 4
        else:
            i += 1
    return entries

def synthesize(entries):
    for i, entry in enumerate(tqdm(entries, desc="🔊 Generating TTS")):
        text = entry["text"]
        output_path = f"tts_segments/segment_{i+1:04d}.wav"

        try:
            inputs = processor(text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(**inputs)
            audio = output.cpu().numpy().squeeze()
            sf.write(output_path, audio, 16000)
        except Exception as e:
            print(f"❌ Error in segment {i+1}: {e}")

if __name__ == "__main__":
    srt_path = "sample_output_translated_ta.srt"  # Replace with your actual path
    entries = parse_srt(srt_path)

    print("🗣️ Parsed", len(entries), "entries from SRT.")
    synthesize(entries)

    print("✅ All segments saved in: tts_segments/")
