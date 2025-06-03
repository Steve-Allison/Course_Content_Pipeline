import logging
import datetime
from pathlib import Path
from course_compiler.config import INPUT_ROOT, OUTPUT_ROOT
from course_compiler.file_utils import sanitize_filename, ensure_dirs

# Set up logging
Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
logfile = Path(OUTPUT_ROOT) / f"run_transcribe_media_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    filename=logfile,
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO
)

# Try to import whisper with graceful fallback
try:
    import whisper
    WHISPER_AVAILABLE = True
    logging.info("Whisper library available for transcription")
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None
    logging.warning("Whisper library not installed. Transcription will be skipped.")
    print("WARNING: Whisper not installed. Install with: pip install openai-whisper")

def transcribe_all_media():
    """Transcribe media files that don't have corresponding caption files."""
    if not WHISPER_AVAILABLE:
        print("Skipping transcription - Whisper not available")
        logging.warning("Transcription skipped due to missing Whisper library")
        return

    input_root = Path(INPUT_ROOT)
    out_dir = Path(OUTPUT_ROOT) / "transcripts"
    ensure_dirs(OUTPUT_ROOT, ["transcripts"])

    # Try to load Whisper model with error handling
    try:
        print("Loading Whisper model (this may take a while on first run)...")
        model = whisper.load_model("base")  # Use base model instead of large for better compatibility
        logging.info("Whisper model loaded successfully")
    except Exception as e:
        print(f"Failed to load Whisper model: {e}")
        logging.error(f"Failed to load Whisper model: {e}")
        print("Try installing with: pip install openai-whisper")
        return

    # Gather all VTT and SRT stems in the input folder
    vtt_srt_stems = {p.stem for p in input_root.glob("*.vtt")}
    vtt_srt_stems.update({p.stem for p in input_root.glob("*.srt")})

    processed_count = 0
    skipped_count = 0

    for media in input_root.rglob("*"):
        if media.suffix.lower() in [".mp4", ".mov", ".mp3", ".wav", ".m4a", ".avi", ".mkv"]:
            if media.stem in vtt_srt_stems:
                print(f"SKIPPING {media.name} (found matching caption file)")
                logging.info(f"Skipped {media.name} - caption file exists")
                skipped_count += 1
                continue

            # Sanitize output filename
            safe_stem = sanitize_filename(media.stem)
            transcript_path = out_dir / (safe_stem + ".txt")

            if transcript_path.exists():
                print(f"Transcript already exists for {media.name}")
                logging.info(f"Skipped {media.name} - transcript exists")
                skipped_count += 1
                continue

            try:
                print(f"Transcribing {media.name}...")
                logging.info(f"Starting transcription of {media.name}")
                result = model.transcribe(str(media))

                with open(transcript_path, "w", encoding='utf-8') as f:
                    f.write(result["text"])

                print(f"✅ Wrote transcript: {transcript_path.name}")
                logging.info(f"Successfully transcribed {media.name} -> {transcript_path}")
                processed_count += 1

            except Exception as e:
                print(f"❌ Failed to transcribe {media.name}: {e}")
                logging.error(f"Failed to transcribe {media.name}: {e}")

    print(f"\nTranscription complete:")
    print(f"  📝 Processed: {processed_count} files")
    print(f"  ⏭️  Skipped: {skipped_count} files")
    print(f"  📄 Transcripts saved to: {out_dir}")

    logging.info(f"Transcription complete. Processed: {processed_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    transcribe_all_media()
