import os
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm import tqdm
from huggingface_hub import login
from langdetect import detect

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Hugging Face token
with open("HUGGINGFACE_TOKEN.txt") as f:
    token = f.read().strip()
login(token)

# Model and tokenizer
MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)

# Target language
TGT_LANG = "ta"  # Tamil

# Supported source language mapping (from langdetect to model tags)
LANG_MAP = {
    "en": "en",
    "hi": "hi",
    "ur": "ur"
}

# SRT timestamp pattern
TIMESTAMP_PATTERN = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})")

def detect_src_lang(text):
    try:
        lang_code = detect(text)
        return LANG_MAP.get(lang_code, "en")  # default to English if unknown
    except Exception:
        return "en"

def translate(text, tgt_lang=TGT_LANG):
    """Translate a single line using IndicTrans2 with auto source language detection."""
    src_lang = detect_src_lang(text)

    # CORRECT order: <src_lang> <2tgt_lang>
    tagged_text = f"<{src_lang}> <2{tgt_lang}> {text}"

    inputs = tokenizer(
        tagged_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(DEVICE)

    generated_tokens = model.generate(
        **inputs,
        max_length=512
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


def translate_srt(input_srt_path, output_srt_path):
    """Translate subtitles from an SRT file and preserve structure."""
    with open(input_srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    translated_lines = []
    buffer = []
    for line in tqdm(lines, desc="Translating"):
        line = line.strip()
        if line.isdigit() or TIMESTAMP_PATTERN.match(line) or line == "":
            if buffer:
                original = " ".join(buffer)
                try:
                    translated = translate(original)
                except Exception as e:
                    translated = "[Translation failed]"
                    print(f"⚠️ Error translating: {original}\n{e}")
                translated_lines.append(translated)
                buffer = []
            translated_lines.append(line)
        else:
            buffer.append(line)

    if buffer:
        original = " ".join(buffer)
        try:
            translated = translate(original)
        except Exception as e:
            translated = "[Translation failed]"
            print(f"⚠️ Error translating: {original}\n{e}")
        translated_lines.append(translated)

    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(translated_lines))

    print(f"✅ Translated SRT saved at: {output_srt_path}")

if __name__ == "__main__":
    input_srt = "sample_output.srt"  # Replace with your actual SRT filename
    output_srt = "sample_output_translated_ta.srt"
    translate_srt(input_srt, output_srt)
