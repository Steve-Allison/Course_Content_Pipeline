import logging
import datetime
from pathlib import Path
from course_compiler.config import INSTRUCTIONAL_JSON_DIR, SUMMARY_DIR, LOGS_DIR
from course_compiler.file_utils import sanitize_filename
from course_compiler.summarisation import (
    flatten_instructional_json,
    deduplicate_glossary,
    write_summary_jsonl,
    write_summary_csv,
    write_markdown_glossary
)

# Ensure output dir exists before logging
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
logfile = Path(LOGS_DIR) / f"summarisation_run_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    filename=logfile,
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO
)

def main():
    print("=== Summarisation/Glossary batch run ===")
    logging.info("Starting summarisation batch run...")

    extracted_dir = Path(INSTRUCTIONAL_JSON_DIR)
    summary_dir = Path(SUMMARY_DIR)
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Gather all extracted JSONs
    extracted_files = list(extracted_dir.glob("*.json"))
    if not extracted_files:
        print("No extracted instructional JSON files found.")
        logging.warning("No extracted instructional JSON files found.")
        return

    # Flatten all instructional JSONs
    extracted_data = []
    for file in extracted_files:
        with open(file, "r") as f:
            content = f.read()
            try:
                import json
                content = json.loads(content)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
        flat = flatten_instructional_json(content, source_file=file.name)
        extracted_data.extend(flat)
    print(f"Loaded {len(extracted_data)} extracted data records.")

    # Write JSONL and CSV summaries
    summary_jsonl = summary_dir / "instructional_summary.jsonl"
    summary_csv = summary_dir / "instructional_summary.csv"
    write_summary_jsonl(extracted_data, summary_jsonl)
    write_summary_csv(extracted_data, summary_csv)
    print(f"Written summary files: {summary_jsonl} and {summary_csv}")

    # Build glossary from all content
    glossary_data = deduplicate_glossary(extracted_data)
    summary_glossary_json = summary_dir / "summary_glossary.json"
    summary_glossary_md = summary_dir / "summary_glossary.md"

    import json
    with open(summary_glossary_json, "w") as f:
        json.dump(glossary_data, f, indent=2)
    write_markdown_glossary(glossary_data, summary_glossary_md)
    print(f"Written glossary files: {summary_glossary_json} and {summary_glossary_md}")

    logging.info(f"Summarisation complete. Files in {summary_dir}")

if __name__ == "__main__":
    main()
