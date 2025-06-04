import json
from pathlib import Path
from course_compiler.config import MEDIA_METADATA_DIR, AUDIO_PREPPED_DIR, VIDEO_PREPPED_DIR
from course_compiler.metadata import ffprobe_metadata
from course_compiler.file_utils import ensure_dirs

def main():
    # Prepare output directory
    output_dir = Path(MEDIA_METADATA_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Directories to scan for prepped media
    prepped_dirs = [
        Path(AUDIO_PREPPED_DIR),
        Path(VIDEO_PREPPED_DIR)
    ]

    found_files = False
    for media_dir in prepped_dirs:
        if not media_dir.exists():
            continue
        for media_file in media_dir.rglob("*.*"):
            found_files = True
        try:
            meta = ffprobe_metadata(str(media_file))
            out_file = output_dir / f"{media_file.stem}_metadata.json"
            with open(out_file, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Wrote metadata for {media_file.name} → {out_file.name}")
        except Exception as e:
            print(f"Failed to extract metadata for {media_file}: {e}")
    if not found_files:
        print("No prepped media files found in audio_prepped or video_prepped.")

if __name__ == "__main__":
    main()
