import whisper
from pathlib import Path
from course_compiler.config import INPUT_ROOT, OUTPUT_ROOT
from course_compiler.file_utils import sanitize_filename, ensure_dirs

def transcribe_all_media():
    model = whisper.load_model("large")  # or "medium", "large", "base"
    input_root = Path(INPUT_ROOT)
    out_dir = Path(OUTPUT_ROOT) / "transcripts"
    ensure_dirs(OUTPUT_ROOT, ["transcripts"])

    # Gather all VTT and SRT stems in the input folder
    vtt_srt_stems = {p.stem for p in input_root.glob("*.vtt")}
    vtt_srt_stems.update({p.stem for p in input_root.glob("*.srt")})

    for media in input_root.glob("*"):
        if media.suffix.lower() in [".mp4", ".mov", ".mp3", ".wav"]:
            if media.stem in vtt_srt_stems:
                print(f"SKIPPING {media.name} (found matching caption file)")
                continue
            # Sanitize output filename
            safe_stem = sanitize_filename(media.stem)
            transcript_path = out_dir / (safe_stem + ".txt")
            if transcript_path.exists():
                print(f"Transcript already exists for {media.name}")
                continue
            print(f"Transcribing {media.name} ...")
            result = model.transcribe(str(media))
            with open(transcript_path, "w") as f:
                f.write(result["text"])
            print(f"Wrote transcript: {transcript_path}")

if __name__ == "__main__":
    transcribe_all_media()
