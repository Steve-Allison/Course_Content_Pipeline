import json
from pathlib import Path
from course_compiler.config import INPUT_ROOT, OUTPUT_ROOT
from course_compiler.metadata import ffprobe_metadata
from course_compiler.file_utils import ensure_dirs

def main():
    # Prepare input/output directories
    media_input = Path(INPUT_ROOT) / "media"
    output_dir = Path(OUTPUT_ROOT) / "media_metadata"
    ensure_dirs(OUTPUT_ROOT, ["media_metadata"])

    # Scan all media files and dump metadata
    for media_file in media_input.rglob("*.*"):
        try:
            meta = ffprobe_metadata(str(media_file))
            out_file = output_dir / f"{media_file.stem}_metadata.json"
            with open(out_file, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Wrote metadata for {media_file.name} → {out_file.name}")
        except Exception as e:
            print(f"Failed to extract metadata for {media_file}: {e}")

if __name__ == "__main__":
    main()
