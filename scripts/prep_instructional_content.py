import logging
import datetime
from pathlib import Path
from course_compiler.config import INPUT_ROOT, OUTPUT_ROOT
from course_compiler.file_utils import sanitize_filename
from course_compiler.extractor import (
    extract_pptx_hierarchical, extract_docx_hierarchical, extract_pdf_hierarchical, save_instructional_json
)

# Ensure output directory exists before logging!
Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
logfile = Path(OUTPUT_ROOT) / f"prep_instructional_content_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    filename=logfile,
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO
)

def main():
    ensure_dirs(OUTPUT_ROOT, ["instructional_json"])
    input_root = Path(INPUT_ROOT)
    out_dir = Path(OUTPUT_ROOT) / "instructional_json"

    all_files = list(input_root.rglob("*"))
    for file in all_files:
        try:
            # Determine output filename (always sanitized)
            safe_filename = sanitize_filename(file.stem) + ".json"
            output_path = out_dir / safe_filename

            if file.suffix.lower() == ".pptx":
                data = extract_pptx_hierarchical(file, out_dir)
                save_instructional_json(data, out_dir, output_filename=safe_filename)
                logging.info(f"SUCCESS: Processed PPTX {file} -> {output_path}")
            elif file.suffix.lower() == ".docx":
                data = extract_docx_hierarchical(file)
                save_instructional_json(data, out_dir, output_filename=safe_filename)
                logging.info(f"SUCCESS: Processed DOCX {file} -> {output_path}")
            elif file.suffix.lower() == ".pdf":
                data = extract_pdf_hierarchical(file, out_dir)
                save_instructional_json(data, out_dir, output_filename=safe_filename)
                logging.info(f"SUCCESS: Processed PDF {file} -> {output_path}")
        except Exception as e:
            logging.error(f"ERROR processing file {file}: {e}", exc_info=True)
    logging.info(f"Instructional extraction complete. Output directory: {out_dir}")

if __name__ == "__main__":
    main()


# This script processes instructional content from various document formats
# and saves the extracted data in a structured JSON format.
# It handles PowerPoint, Word, and PDF files, extracting text and images,
# and organizes the content hierarchically for easier analysis and use in educational contexts.
# It ensures the output directory exists and saves the results in a specified folder.
# The logging captures all actions, successes, and errors for later review.
# This script is part of a larger course content preparation pipeline.