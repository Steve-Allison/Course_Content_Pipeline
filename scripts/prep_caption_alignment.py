import logging
import datetime
from pathlib import Path
from course_compiler.config import INPUT_ROOT, OUTPUT_ROOT
from course_compiler.file_utils import sanitize_filename, ensure_dirs
from course_compiler.vtt_srt_prep import (
    parse_vtt_segments, parse_srt_segments,
    clean_caption_text, semantic_process_segments, save_caption_segments
)
import sys

Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
logfile = Path(OUTPUT_ROOT) / f"prep_caption_alignment_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    filename=logfile,
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO
)

def main():
    input_root = Path(INPUT_ROOT)
    out_dir = Path(OUTPUT_ROOT) / "caption_prepped"
    ensure_dirs(OUTPUT_ROOT, ["caption_prepped"])

    vtt_files = list(input_root.rglob("*.vtt"))
    srt_files = list(input_root.rglob("*.srt"))
    total_files = len(vtt_files) + len(srt_files)
    logging.info(f"Found {len(vtt_files)} VTT files and {len(srt_files)} SRT files.")

    if total_files == 0:
        print("No VTT or SRT caption files found. Skipping caption alignment step.")
        logging.warning("No VTT or SRT caption files found. Caption alignment step skipped.")
        # Optionally, create a marker file so downstream knows this was deliberate
        marker = out_dir / ".no_captions_found"
        marker.touch()
        sys.exit(0)

    for vtt_file in vtt_files:
        try:
            segments = parse_vtt_segments(vtt_file)
            cleaned_segments = clean_caption_text(segments)
            sem_segments = semantic_process_segments(cleaned_segments, use_llm=True)
            safe_stem = sanitize_filename(vtt_file.stem)
            output_json = out_dir / (safe_stem + "_vtt_semantic.json")
            save_caption_segments(sem_segments, output_json)
            logging.info(f"Processed VTT: {vtt_file.name} → {output_json.name}")
        except Exception as e:
            logging.error(f"Error processing VTT {vtt_file}: {e}", exc_info=True)

    for srt_file in srt_files:
        try:
            segments = parse_srt_segments(srt_file)
            cleaned_segments = clean_caption_text(segments)
            sem_segments = semantic_process_segments(cleaned_segments, use_llm=True)
            safe_stem = sanitize_filename(srt_file.stem)
            output_json = out_dir / (safe_stem + "_srt_semantic.json")
            save_caption_segments(sem_segments, output_json)
            logging.info(f"Processed SRT: {srt_file.name} → {output_json.name}")
        except Exception as e:
            logging.error(f"Error processing SRT {srt_file}: {e}", exc_info=True)

if __name__ == "__main__":
    main()
# This script processes caption files in VTT and SRT formats, cleaning and semantically processing them.
# It saves the processed segments in a JSON format for further use.
# It also logs the processing steps and any errors encountered.
# The script ensures that the output directory exists and handles cases where no caption files are found.
# It uses a marker file to indicate that no captions were found, which can be useful for downstream processes.
# The script is designed to be run as a standalone module, processing all caption files found in the specified input directory.
# It uses the course_compiler library for file handling and caption processing.