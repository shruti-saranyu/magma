import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import srt
from tqdm import tqdm
import datetime
from langdetect import detect

# === CONFIGURATION ===
MODEL_NAME = "ai4bharat/indictrans2-indic-indic-1B"
SOURCE_SRT_PATH = "sample_output_speakers.srt"
TARGET_SRT_PATH = "sample_output_translated_ta.srt"
TARGET_LANG_TAG = "<2ta>"  # Tamil
HUGGINGFACE_TOKEN_PATH = "HUGGINGFACE_TOKEN.txt"

# === LOAD TOKEN ===
with open(HUGGINGFACE_TOKEN_PATH, "r") as f:
    hf_token = f.read().strip()

# === LOAD TOKENIZER & MODEL ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token=hf_token, trust_remote_code=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === HELPERS ===
def detect_lang(text):
    try:
        return detect(text)
    except:
        return "en"

def add_tags(text, source_lang):
    if not text.strip():
        return text
    return f"<{source_lang}> {TARGET_LANG_TAG} {text}"

def translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

# === PROCESS SRT ===
with open(SOURCE_SRT_PATH, "r", encoding="utf-8") as f:
    original_subs = list(srt.parse(f.read()))

translated_subs = []

for sub in tqdm(original_subs, desc="Translating"):
    raw_text = sub.content.strip()
    
    if not raw_text:
        translated_subs.append(sub)
        continue

    # Detect language for tagging
    lang = detect_lang(raw_text)
    if lang == "ur":
        source_lang_tag = "ur"
    elif lang == "hi":
        source_lang_tag = "hi"
    elif lang == "en":
        source_lang_tag = "en"
    else:
        source_lang_tag = "hi"  # fallback

    try:
        tagged_text = add_tags(raw_text, source_lang_tag)
        translated_text = translate(tagged_text)
        sub.content = translated_text
    except Exception as e:
        print(f"⚠️ Error translating: {sub.start} --> {sub.end} {sub.content}")
        print(f"Reason: {e}")
    translated_subs.append(sub)

# === WRITE TRANSLATED SRT ===
with open(TARGET_SRT_PATH, "w", encoding="utf-8") as f:
    f.write(srt.compose(translated_subs))

print(f"✅ Translated SRT saved at: {TARGET_SRT_PATH}")
