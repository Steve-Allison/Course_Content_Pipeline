import json
import csv
from pathlib import Path
from collections import defaultdict

def load_json_records(files):
    """
    Load and aggregate lists of JSON records from a list of files.
    Returns a flat list of all items.
    """
    data = []
    for file in files:
        try:
            with open(file, "r") as f:
                content = json.load(f)
                if isinstance(content, list):
                    data.extend(content)
                elif isinstance(content, dict):
                    data.append(content)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return data

def write_summary_jsonl(data, path):
    """
    Write data records to a .jsonl file (one JSON object per line).
    """
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def write_summary_csv(data, path):
    """
    Write data records to a .csv file.
    """
    if not data:
        print(f"No data to write for {path}")
        return
    keys = set()
    for item in data:
        keys.update(item.keys())
    keys = sorted(keys)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for item in data:
            writer.writerow({k: item.get(k, "") for k in keys})

def deduplicate_glossary(data):
    """
    Create a glossary of unique terms from all keywords found in data.
    Counts frequency, sources, and provides up to 3 example contexts.
    """
    glossary_map = defaultdict(lambda: {
        "occurrences": 0,
        "sources": set(),
        "sample_contexts": set()
    })
    for item in data:
        source = item.get("source_file", "unknown")
        text = item.get("text", "")
        for kw in item.get("keywords", []):
            term = kw[0] if isinstance(kw, (list, tuple)) else kw
            entry = glossary_map[term]
            entry["occurrences"] += 1
            entry["sources"].add(source)
            if len(entry["sample_contexts"]) < 3:
                entry["sample_contexts"].add(text)
    result = []
    for term, info in glossary_map.items():
        result.append({
            "term": term,
            "occurrences": info["occurrences"],
            "sources": sorted(list(info["sources"])),
            "sample_contexts": sorted(list(info["sample_contexts"]))
        })
    return result

def write_markdown_glossary(glossary_data, output_md_path):
    """
    Export glossary data as a Markdown file for human review.
    """
    with open(output_md_path, "w") as f:
        f.write("# 📘 Glossary Summary\n\n")
        for item in sorted(glossary_data, key=lambda x: -x["occurrences"]):
            f.write(f"## {item['term']}\n")
            f.write(f"- **Occurrences:** {item['occurrences']}\n")
            f.write(f"- **Sources:** {', '.join(item['sources'])}\n")
            f.write(f"- **Sample Contexts:**\n")
            for ctx in item['sample_contexts']:
                f.write(f"  - {ctx.strip()}\n")
            f.write("\n")

def flatten_instructional_json(data, source_file=None):
    flat = []
    if isinstance(data, dict):
        # If this dict represents a slide/page/section/component, capture it
        if any(k in data for k in ["slide_number", "page_number", "section", "heading", "text", "components"]):
            item = dict(data)
            if source_file:
                item["source_file"] = source_file
            flat.append(item)
        for v in data.values():
            flat.extend(flatten_instructional_json(v, source_file=source_file))
    elif isinstance(data, list):
        for v in data:
            flat.extend(flatten_instructional_json(v, source_file=source_file))
    return flat
