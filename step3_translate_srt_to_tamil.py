from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import pysrt

# Load HF token
with open("HUGGINGFACE_TOKEN.txt") as f:
    HF_TOKEN = f.read().strip()

# Model name (correct version)
MODEL_NAME = "ai4bharat/indictrans2-indic-indic-1B"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, token=HF_TOKEN, trust_remote_code=True)

# Translation pipeline
translation_pipeline = pipeline(
    "translation",
    model=model,
    tokenizer=tokenizer,
    src_lang="indic",  # this is required but is overridden by tags like <hi>, <ur>
    tgt_lang="ta",
    max_length=512,
)

# Input SRT with speaker tags
SOURCE_SRT_PATH = "sample_output.srt"
# Output SRT in Tamil
TARGET_SRT_PATH = "sample_output_translated_ta.srt"

# Load the SRT subtitles
subs = pysrt.open(SOURCE_SRT_PATH)

translated_subs = []

# Translation loop
for sub in subs:
    text = sub.text.strip()
    if not text:
        translated_subs.append("")
        continue

    # Guess language tag (assume Hindi unless Urdu characters detected)
    if any("\u0600" <= c <= "\u06FF" for c in text):
        lang_tag = "<ur>"
    else:
        lang_tag = "<hi>"

    # Translation prompt with IndicTrans-style tags
    input_text = f"{lang_tag} <2ta> {text}"

    try:
        translated = translation_pipeline(input_text)[0]["translation_text"]
        translated_subs.append(translated)
    except Exception as e:
        print(f"⚠️ Error translating: {sub.start} --> {sub.end} {text}")
        print(e)
        translated_subs.append("")  # Keep blank if failed

# Save translated subtitles
for sub, translated_text in zip(subs, translated_subs):
    sub.text = translated_text

subs.save(TARGET_SRT_PATH, encoding="utf-8")
print(f"✅ Translated SRT saved at: {TARGET_SRT_PATH}")
