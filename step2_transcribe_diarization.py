import os
import ffmpeg
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from pyannote.core import Segment
from datetime import timedelta


def load_hf_token_from_file(file_path="hf_token.txt"):
    """Read Hugging Face token from a local file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Token file '{file_path}' not found.")
    with open(file_path, "r") as f:
        return f.read().strip()


def extract_audio(video_path, audio_path):
    """Extract mono, 16kHz audio from a video file using ffmpeg."""
    ffmpeg.input(video_path).output(audio_path, ac=1, ar='16000').run(overwrite_output=True)


def transcribe_with_diarization(video_path, hf_token=None):
    """Transcribe the video with speaker diarization and return SRT file path."""
    audio_path = os.path.splitext(video_path)[0] + "_audio.wav"

    # Step 1: Extract audio
    print("🎵 Extracting audio...")
    extract_audio(video_path, audio_path)

    # Step 2: Load Hugging Face token
    if not hf_token:
        hf_token = load_hf_token_from_file("hf_token.txt")

    # Step 3: Load diarization model
    print("🧑‍🤝‍🧑 Performing diarization...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token
    )
    diarization = pipeline(audio_path)

    # Step 4: Transcribe using Faster-Whisper
    print("🧠 Transcribing with Whisper...")
    model = WhisperModel("medium", compute_type="float32")
    segments, _ = model.transcribe(audio_path, beam_size=5, language="ur")

    # Step 5: Generate SRT with speaker labels
    srt_output = []
    seg_index = 1

    for segment in segments:
        whisper_segment = Segment(segment.start, segment.end)

        # Match speaker
        speaker_label = "Unknown"
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.intersects(whisper_segment):
                speaker_label = speaker
                break

        # Format time
        start_time = str(timedelta(seconds=int(segment.start)))
        end_time = str(timedelta(seconds=int(segment.end)))

        # Add to SRT
        srt_output.append(f"{seg_index}")
        srt_output.append(f"{start_time.replace('.', ',')} --> {end_time.replace('.', ',')}")
        srt_output.append(f"{speaker_label}: {segment.text}")
        srt_output.append("")
        seg_index += 1

    # Write to file
    srt_path = os.path.splitext(video_path)[0] + "_output.srt"
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        srt_file.write("\n".join(srt_output))

    print(f"✅ Transcription with diarization saved to: {srt_path}")
    return srt_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python step2_transcribe_diarization.py <input_video>")
        sys.exit(1)

    input_video = sys.argv[1]
    huggingface_token = None  # Leave as None to read from file
    output_srt = transcribe_with_diarization(input_video, huggingface_token)
