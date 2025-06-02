import os
import glob
import json
from collections import Counter
import argparse
from course_compiler.logging_utils import setup_logger
logger = setup_logger(__name__)

def load_segments_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    elif isinstance(data, list):
        return data
    return []

def count_source_tags(json_files):
    counter = Counter()
    for fpath in json_files:
        for segment in load_segments_from_file(fpath):
            tag = segment.get("metadata", {}).get("source_tag", "unknown")
            counter[tag] += 1
    return counter

def main():
    parser = argparse.ArgumentParser(description="Count segment source tags in output directory.")
    parser.add_argument("--output_dir", default="output", help="Path to directory containing content JSON files")
    parser.add_argument("--output_file", default="segment_counts_by_source.json", help="Filename for output JSON")
    args = parser.parse_args()

    json_files = glob.glob(os.path.join(args.output_dir, "*.json")) + glob.glob(os.path.join(args.output_dir, "captions", "*.json"))
    counts = count_source_tags(json_files)
    logger.info("Segment counts by source_tag:")
    for tag, count in counts.items():
        logger.info(f"{tag:15}: {count}")
    output_path = os.path.join(args.output_dir, args.output_file)
    with open(output_path, "w") as f:
        json.dump(counts, f, indent=2)
    logger.info("Saved source tag counts to %s", output_path)

if __name__ == "__main__":
    main()