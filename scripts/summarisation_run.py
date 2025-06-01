import logging
import datetime
from pathlib import Path
from course_compiler.config import OUTPUT_ROOT
from course_compiler.file_utils import sanitize_filename
from course_compiler.summarisation import (
    load_json_records,
    deduplicate_glossary,
    write_summary_jsonl,
    write_summary_csv,
    write_markdown_glossary
)

# Ensure output dir exists before logging
Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
logfile = Path(OUTPUT_ROOT) / f"summarisation_run_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    filename=logfile,
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO
)

def main():
    print("=== Summarisation/Glossary batch run ===")
    logging.info("Starting summarisation batch run...")

    extracted_dir = Path(OUTPUT_ROOT) / "instructional_json"
    summary_dir = Path(OUTPUT_ROOT) / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Gather all extracted JSONs
    extracted_files = list(extracted_dir.glob("*.json"))
    if not extracted_files:
        print("No extracted instructional JSON files found.")
        logging.warning("No extracted instructional JSON files found.")
        return

    extracted_data = load_json_records(extracted_files)
    print(f"Loaded {len(extracted_data)} extracted data records.")

    # Write JSONL and CSV summaries
    summary_jsonl = summary_dir / "instructional_summary.jsonl"
    summary_csv = summary_dir / "instructional_summary.csv"
    write_summary_jsonl(extracted_data, summary_jsonl)
    write_summary_csv(extracted_data, summary_csv)
    logging.info(f"Saved instructional summary to {summary_jsonl} and {summary_csv}")

    # Deduplicate glossary and export
    glossary = deduplicate_glossary(extracted_data)
    glossary_json = summary_dir / "summary_glossary.json"
    glossary_md = summary_dir / "summary_glossary.md"
    with open(glossary_json, "w") as f:
        import json
        json.dump(glossary, f, indent=2)
    write_markdown_glossary(glossary, glossary_md)
    print(f"Wrote glossary: {glossary_json} and {glossary_md}")
    logging.info(f"Wrote glossary: {glossary_json} and {glossary_md}")

    # Example: If you ever generate per-topic summaries
    # for topic in topics:
    #     safe_topic = sanitize_filename(topic)
    #     out_path = summary_dir / f"{safe_topic}_summary.json"
    #     with open(out_path, "w") as f:
    #         json.dump(data, f)

    print("Summarisation/Glossary batch complete.")
    logging.info("Summarisation batch complete.")

if __name__ == "__main__":
    main()
