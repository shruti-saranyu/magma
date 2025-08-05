import os
import re
import torch
from tqdm import tqdm
import soundfile as sf
from transformers import AutoProcessor, AutoModelForTextToSpeech

# 📦 Load IndicParler-TTS model and processor
print("📦 Loading IndicParler-TTS model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("ai4bharat/indic-parler-tts", trust_remote_code=True)
model = AutoModelForTextToSpeech.from_pretrained("ai4bharat/indic-parler-tts", trust_remote_code=True).to(device)

# Tamil male emotional prompts
EMOTIONAL_MALE_PROMPTS = [
    "a calm Tamil-speaking male",
    "an excited Tamil-speaking young male",
    "a sad Tamil-speaking adult male",
    "an angry Tamil-speaking male",
    "a happy Tamil-speaking boy"
]

# Regex to extract speaker label
SPEAKER_PATTERN = re.compile(r"^Speaker\s+(\w+):\s*(.+)$")

# Output directory
os.makedirs("tts_segments", exist_ok=True)

def parse_srt(srt_path):
    """Parse translated SRT and return list of dicts with start, end, speaker, text."""
    with open(srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries = []
    i = 0
    while i < len(lines):
        if lines[i].strip().isdigit():
            start_end = lines[i+1].strip()
            text_line = lines[i+2].strip()
            match = SPEAKER_PATTERN.match(text_line)

            if match:
                speaker = match.group(1)
                text = match.group(2)
            else:
                speaker = "Unknown"
                text = text_line

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
    """Assign different emotional prompts to each speaker."""
    mapping = {}
    idx = 0
    for spk in sorted(set(speaker_list)):
        mapping[spk] = EMOTIONAL_MALE_PROMPTS[idx % len(EMOTIONAL_MALE_PROMPTS)]
        idx += 1
    return mapping

def synthesize(entries, speaker_mapping):
    """Generate audio segments using IndicParler-TTS and save them."""
    for i, entry in enumerate(tqdm(entries, desc="🔊 Generating TTS")):
        description = speaker_mapping.get(entry["speaker"], EMOTIONAL_MALE_PROMPTS[0])
        text = entry["text"]
        output_path = f"tts_segments/segment_{i+1:04d}.wav"

        try:
            inputs = processor(
                text=text,
                description=description,
                speaker_id="ta_male",
                language="ta",
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            audio = outputs.audio[0].cpu().numpy()
            sf.write(output_path, audio, 16000)
        except Exception as e:
            print(f"❌ Error generating TTS for segment {i+1}: {e}")

if __name__ == "__main__":
    srt_path = "sample_output_translated_ta.srt"
    entries = parse_srt(srt_path)

    speakers = [e["speaker"] for e in entries]
    speaker_map = assign_emotional_prompts(speakers)

    print("🗣️ Speaker Prompt Mapping:")
    for k, v in speaker_map.items():
        print(f"  Speaker {k} → \"{v}\"")

    synthesize(entries, speaker_map)

    print("✅ TTS generation completed with emotional Tamil voices. Files saved in: tts_segments/")
