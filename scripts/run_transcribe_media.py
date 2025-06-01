import logging
from pathlib import Path
import os

from course_compiler.config import INPUT_ROOT, OUTPUT_ROOT
from course_compiler.file_utils import ensure_dirs, sanitize_filename
from course_compiler.vtt_srt_prep import get_vtt_srt_stems
from course_compiler.extractor import transcribe_file

def transcribe_all_media():
    input_root = Path(INPUT_ROOT) / "media"
    out_dir = Path(OUTPUT_ROOT) / "transcripts"
    ensure_dirs(OUTPUT_ROOT, ["transcripts"])

    # Gather stems for which VTT/SRT captions already exist
    vtt_srt_dir = Path(INPUT_ROOT) / "captions"
    vtt_srt_stems = get_vtt_srt_stems(vtt_srt_dir)

    for media in input_root.iterdir():
        # Only process common media extensions
        if media.suffix.lower() not in [".mp4", ".mov", ".mp3", ".wav"]:
            continue

        # Skip if a matching caption file exists
        if media.stem in vtt_srt_stems:
            print(f"Skipping {media.name} – captions already provided")
            continue

        safe_stem = sanitize_filename(media.stem)
        transcript_path = out_dir / (safe_stem + ".txt")
        if transcript_path.exists():
            print(f"Transcript already exists for {media.name}")
            continue

        try:
            print(f"Transcribing {media.name}...")
            transcription_text = transcribe_file(str(media))
            with open(transcript_path, "w") as f:
                f.write(transcription_text)
            print(f"Wrote transcript for {media.name} → {transcript_path.name}")
        except Exception as e:
            print(f"Error transcribing {media.name}: {e}")

if __name__ == "__main__":
    transcribe_all_media()