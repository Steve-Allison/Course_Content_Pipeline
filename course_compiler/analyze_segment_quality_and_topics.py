

import os
import glob
import json
import argparse
from collections import Counter
from course_compiler.logging_utils import setup_logger

logger = setup_logger(__name__)


# Thresholds for warnings
TOO_SHORT_THRESHOLD = 0.1
MISSING_SUMMARY_THRESHOLD = 0.2
MISSING_TOPICS_THRESHOLD = 0.2
MISSING_TAGS_THRESHOLD = 0.2

def load_segments_from_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "segments" in data:
        return data["segments"]
    elif isinstance(data, list):
        return data
    return []

def analyze_segment_quality(json_files):
    too_short = 0
    missing_summary = 0
    missing_topics = 0
    missing_tags = 0
    total = 0

    for path in json_files:
        for seg in load_segments_from_file(path):
            text = seg.get("text", "")
            total += 1
            if len(text.split()) < 5:
                too_short += 1
            if not seg.get("summary"):
                missing_summary += 1
            if not seg.get("topics"):
                missing_topics += 1
            if not seg.get("tags"):
                missing_tags += 1

    return {
        "total_segments": total,
        "too_short": too_short,
        "missing_summary": missing_summary,
        "missing_topics": missing_topics,
        "missing_tags": missing_tags
    }

def analyze_topic_coverage(json_files):
    all_topics = []
    with_topic = 0
    without_topic = 0

    for path in json_files:
        for seg in load_segments_from_file(path):
            topics = seg.get("topics", [])
            if topics:
                with_topic += 1
                all_topics.extend(topics)
            else:
                without_topic += 1

    return {
        "segments_with_topics": with_topic,
        "segments_without_topics": without_topic,
        "total_topics": len(set(all_topics)),
        "topic_frequency": Counter(all_topics)
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze segment quality and topic coverage.")
    parser.add_argument("--output_dir", default="output", help="Directory with content and caption JSON files.")
    parser.add_argument("--quality_file", default="segment_quality_report.json", help="Filename for quality report.")
    parser.add_argument("--topic_file", default="topic_coverage_report.json", help="Filename for topic report.")
    parser.add_argument("--warnings_file", default="segment_quality_warnings.json", help="Filename for warning output.")
    args = parser.parse_args()

    json_files = glob.glob(os.path.join(args.output_dir, "*.json")) + glob.glob(os.path.join(args.output_dir, "captions", "*.json"))

    quality_report = analyze_segment_quality(json_files)
    topic_report = analyze_topic_coverage(json_files)

    with open(os.path.join(args.output_dir, args.quality_file), "w") as f:
        json.dump(quality_report, f, indent=2)

    with open(os.path.join(args.output_dir, args.topic_file), "w") as f:
        json.dump(topic_report, f, indent=2)

    logger.info("Wrote %s and %s", args.quality_file, args.topic_file)

    def warn_if_exceeds(label, count, total, threshold):
        ratio = count / total if total > 0 else 0
        if ratio > threshold:
            return True, f"WARNING: {label} is {ratio:.1%}, which exceeds threshold of {threshold:.0%}"
        return False, ""

    warnings = []
    exceeded, msg = warn_if_exceeds("Too short segments", quality_report["too_short"], quality_report["total_segments"], TOO_SHORT_THRESHOLD)
    if exceeded:
        warnings.append(msg)
    exceeded, msg = warn_if_exceeds("Missing summary", quality_report["missing_summary"], quality_report["total_segments"], MISSING_SUMMARY_THRESHOLD)
    if exceeded:
        warnings.append(msg)
    exceeded, msg = warn_if_exceeds("Missing topics", quality_report["missing_topics"], quality_report["total_segments"], MISSING_TOPICS_THRESHOLD)
    if exceeded:
        warnings.append(msg)
    exceeded, msg = warn_if_exceeds("Missing tags", quality_report["missing_tags"], quality_report["total_segments"], MISSING_TAGS_THRESHOLD)
    if exceeded:
        warnings.append(msg)

    with open(os.path.join(args.output_dir, args.warnings_file), "w") as f:
        json.dump(warnings, f, indent=2)

    for w in warnings:
        logger.warning(w)

if __name__ == "__main__":
    main()