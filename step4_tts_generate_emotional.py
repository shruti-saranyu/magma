import os
import re
import torch
import soundfile as sf
from tqdm import tqdm

from parler_tts import ParlerTTS  # Uses Hugging Face package
from parler_tts.audio_utils import load_audio

# 📦 Load IndicParler-TTS model from Hugging Face (no local flash-attn dependency)
print("📦 Loading IndicParler-TTS model from Hugging Face...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = ParlerTTS.from_pretrained("ai4bharat/indic-parler-tts", device=device)

# Tamil emotional speaker descriptions
EMOTIONAL_MALE_PROMPTS = [
    "a calm Tamil-speaking male",
    "an excited Tamil-speaking young male",
    "a sad Tamil-speaking adult male",
    "an angry Tamil-speaking male",
    "a happy Tamil-speaking boy"
]

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

def assign_emotional_prompts(speaker_list):
    mapping = {}
    idx = 0
    for spk in sorted(set(speaker_list)):
        mapping[spk] = EMOTIONAL_MALE_PROMPTS[idx % len(EMOTIONAL_MALE_PROMPTS)]
        idx += 1
    return mapping

def synthesize(entries, speaker_mapping):
    for i, entry in enumerate(tqdm(entries, desc="🔊 Generating TTS")):
        description = speaker_mapping.get(entry["speaker"], EMOTIONAL_MALE_PROMPTS[0])
        text = entry["text"]
        output_path = f"tts_segments/segment_{i+1:04d}.wav"

        try:
            wav = tts.synthesize(
                text=text,
                speaker="ta_male",
                language="ta",
                description=description,
            )
            sf.write(output_path, wav, 16000)
        except Exception as e:
            print(f"❌ Error in segment {i+1}: {e}")

if __name__ == "__main__":
    srt_path = "sample_output_translated_ta.srt"  # Replace with your actual file
    entries = parse_srt(srt_path)

    speakers = [e["speaker"] for e in entries]
    speaker_map = assign_emotional_prompts(speakers)

    print("🗣️ Speaker Prompt Mapping:")
    for k, v in speaker_map.items():
        print(f"  Speaker {k} → \"{v}\"")

    synthesize(entries, speaker_map)

    print("✅ All segments saved in: tts_segments/")
