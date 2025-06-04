import os
import glob
import json
from collections import Counter
import argparse
from course_compiler.logging_utils import setup_logger
logger = setup_logger(__name__)

OUTPUT_DIR = "output"
GLOSSARY_PATH = os.path.join(OUTPUT_DIR, "glossary.json")

def load_segments():
    files = glob.glob(os.path.join(OUTPUT_DIR, "*.json")) + glob.glob(os.path.join(OUTPUT_DIR, "captions", "*.json"))
    segments = []
    for path in files:
        # Don't process the glossary file itself
        if os.path.basename(path) == "glossary.json":
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "segments" in data:
            segments.extend(data["segments"])
        elif isinstance(data, list):
            segments.extend(data)
    return segments

def load_glossary_terms():
    if not os.path.exists(GLOSSARY_PATH):
        return []
    with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [entry["term"] for entry in data if "term" in entry]

def count_glossary_usage(segments, glossary_terms):
    counter = Counter()
    for seg in segments:
        text = seg.get("text", "").lower()
        for term in glossary_terms:
            if term.lower() in text:
                counter[term] += 1
    return counter

def safe_output_path(output_dir, filename):
    """Create output path and ensure directory exists for relative paths."""
    if "/" in filename:
        output_path = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    else:
        output_path = os.path.join(output_dir, filename)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Analyze glossary usage and suggest improvements.")
    parser.add_argument("--output_dir", default="output", help="Directory containing output files")
    parser.add_argument("--usage_file", default="glossary_term_usage.json")
    parser.add_argument("--unused_file", default="unused_glossary_terms.json")
    parser.add_argument("--improvement_file", default="glossary_improvement_prompts.txt")
    parser.add_argument("--ai_prompt_file", default="glossary_ai_prompt_templates.txt")
    args = parser.parse_args()

    segments = load_segments()
    glossary_terms = load_glossary_terms()

    usage_counts = count_glossary_usage(segments, glossary_terms)
    unused_terms = [term for term in glossary_terms if usage_counts.get(term, 0) == 0]

    # Write output files with proper path handling
    usage_path = safe_output_path(args.output_dir, args.usage_file)
    unused_path = safe_output_path(args.output_dir, args.unused_file)
    improvement_path = safe_output_path(args.output_dir, args.improvement_file)
    ai_prompt_path = safe_output_path(args.output_dir, args.ai_prompt_file)

    with open(usage_path, "w", encoding="utf-8") as f:
        json.dump(dict(usage_counts), f, indent=2, ensure_ascii=False)

    with open(unused_path, "w", encoding="utf-8") as f:
        json.dump(unused_terms, f, indent=2, ensure_ascii=False)

    prompt_lines = []
    for term in glossary_terms:
        count = usage_counts.get(term, 0)
        if count == 0:
            prompt_lines.append(f"- Consider removing or revising the glossary term: '{term}' (unused)")
        elif count < 3:
            prompt_lines.append(f"- Consider expanding usage/examples of: '{term}' (only mentioned {count} time{'s' if count != 1 else ''})")

    with open(improvement_path, "w", encoding="utf-8") as f:
        f.write("\n".join(prompt_lines))

    ai_prompt_lines = []
    for term in glossary_terms:
        count = usage_counts.get(term, 0)
        if count == 0:
            ai_prompt_lines.append(f"Please evaluate and either revise or remove the glossary entry for '{term}' as it is not used in the current content.")
        elif count < 3:
            ai_prompt_lines.append(f"'{term}' is used only {count} time(s). Suggest examples, analogies, or more usages to enhance its instructional value.")

    with open(ai_prompt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(ai_prompt_lines))

    logger.info("Glossary usage written to %s", args.usage_file)
    logger.info("Unused glossary terms written to %s", args.unused_file)

if __name__ == "__main__":
    main()
