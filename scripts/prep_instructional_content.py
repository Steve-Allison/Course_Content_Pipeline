import logging
import datetime
from pathlib import Path
from course_compiler.config import INPUT_ROOT, INSTRUCTIONAL_JSON_DIR, IMAGES_DIR, LOGS_DIR
from course_compiler.file_utils import sanitize_filename, ensure_dirs
from course_compiler.extractors import extract_content
import json

# Ensure output directory exists before logging!
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
logfile = Path(LOGS_DIR) / f"prep_instructional_content_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    filename=logfile,
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO
)

def save_extraction_results(data, output_dir, output_filename):
    """Save extracted content as JSON with proper filename sanitization."""
    output_path = Path(output_dir) / output_filename
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return output_path

def main():
    # Ensure directories exist
    Path(INSTRUCTIONAL_JSON_DIR).mkdir(parents=True, exist_ok=True)
    Path(IMAGES_DIR).mkdir(parents=True, exist_ok=True)

    input_root = Path(INPUT_ROOT)
    out_dir = Path(INSTRUCTIONAL_JSON_DIR)
    images_dir = Path(IMAGES_DIR)

    all_files = list(input_root.rglob("*"))
    supported_extensions = {'.pptx', '.ppt', '.docx', '.doc', '.pdf', '.xlsx', '.xlsm', '.xltx', '.xls', '.txt', '.md'}

    processed_count = 0

    for file in all_files:
        if not file.is_file() or file.suffix.lower() not in supported_extensions:
            continue

        try:
            # Determine output filename (always sanitized)
            safe_filename = sanitize_filename(file.stem) + ".json"
            output_path = out_dir / safe_filename

            # Configure extraction with images directory
            config = {
                'images_dir': str(images_dir)
            }

            # Use the new extractors package
            data = extract_content(file, config)

            # Save the results
            save_extraction_results(data, out_dir, safe_filename)

            logging.info(f"SUCCESS: Processed {file.suffix.upper()} {file} -> {output_path}")
            processed_count += 1

        except Exception as e:
            logging.error(f"ERROR processing file {file}: {e}", exc_info=True)

    logging.info(f"Instructional extraction complete. Processed {processed_count} files. Output directory: {out_dir}")
    print(f"Processed {processed_count} instructional files. Check {logfile} for details.")

if __name__ == "__main__":
    main()


# This script processes instructional content from various document formats
# and saves the extracted data in a structured JSON format.
# It handles PowerPoint, Word, and PDF files, extracting text and images,
# and organizes the content hierarchically for easier analysis and use in educational contexts.
# It ensures the output directory exists and saves the results in a specified folder.
# The logging captures all actions, successes, and errors for later review.
# This script is part of a larger course content preparation pipeline.
