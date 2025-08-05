import os
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from tqdm import tqdm
from huggingface_hub import login

# Load IndicTrans2 model and tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ai4bharat/indictrans2-en-ta"  # ✅ Correct
  # Multilingual model
with open("HUGGINGFACE_TOKEN.txt") as f:
    token = f.read().strip()
login(token)  
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Language codes
SRC_LANG = "ur"  # Source language (update this if needed)
TGT_LANG = "ta"  # Target language (Tamil)

# SRT parsing pattern
TIMESTAMP_PATTERN = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})")

def translate(text, src_lang=SRC_LANG, tgt_lang=TGT_LANG):
    """Translate a single line of text using IndicTrans2."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(DEVICE)

    inputs['lang_code'] = torch.tensor([tokenizer.lang_code_to_id[f"{src_lang}_XX"]]).to(DEVICE)
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[f"{tgt_lang}_XX"],
        max_length=512
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def translate_srt(input_srt_path, output_srt_path):
    """Translate the text lines in an SRT file while preserving format."""
    with open(input_srt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    translated_lines = []
    buffer = []
    for line in tqdm(lines, desc="Translating"):
        line = line.strip()
        if line.isdigit() or TIMESTAMP_PATTERN.match(line) or line == "":
            # Flush buffer if any
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
            # Accumulate speaker+text line
            buffer.append(line)

    # Catch any leftover buffer
    if buffer:
        original = " ".join(buffer)
        try:
            translated = translate(original)
        except:
            translated = "[Translation failed]"
        translated_lines.append(translated)

    # Write output
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(translated_lines))

    print(f"✅ Translated SRT saved at: {output_srt_path}")

if __name__ == "__main__":
    input_srt = "sample_output.srt"  # Replace with your actual SRT filename
    output_srt = "sample_output_translated_ta.srt"
    translate_srt(input_srt, output_srt)
