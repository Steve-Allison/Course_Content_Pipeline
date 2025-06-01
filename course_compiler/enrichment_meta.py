import os
import json
import csv

def group_segments_by_topic(topics_summary_path, outdir, filename_sanitizer=None):
    """
    Reads topics_summary.json (a list of caption segments with topics) and
    saves out a file per topic containing all its segments.
    Sanitizes all output filenames if filename_sanitizer is provided.
    """
    if not os.path.exists(topics_summary_path):
        raise FileNotFoundError(f"Could not find {topics_summary_path}")

    with open(topics_summary_path, "r") as f:
        segments = json.load(f)
    # If segments is not a list, fail gracefully
    if not isinstance(segments, list):
        raise ValueError(f"Expected a list of segments in {topics_summary_path}")

    # Group segments by topic_id (or some key)
    topic_groups = {}
    for seg in segments:
        topic = seg.get("topic_id", "no_topic")
        topic_groups.setdefault(topic, []).append(seg)

    os.makedirs(outdir, exist_ok=True)
    written_files = []

    for topic_id, segs in topic_groups.items():
        filename = f"topic_{topic_id}.json"
        if filename_sanitizer:
            filename = filename_sanitizer(filename)
        output_path = os.path.join(outdir, filename)
        with open(output_path, "w") as fout:
            json.dump(segs, fout, indent=2)
        written_files.append(output_path)

    print(f"Wrote {len(written_files)} topic files to {outdir}")

def print_tag_coverage(instr_dir):
    """
    Scans all instructional JSON files in instr_dir and prints a count of tag types (if 'tags' field present).
    """
    tag_counter = {}
    files = [f for f in os.listdir(instr_dir) if f.endswith(".json")]
    for fname in files:
        with open(os.path.join(instr_dir, fname), "r") as f:
            data = json.load(f)
        tags = data.get("tags", [])
        for tag in tags:
            tag_counter[tag] = tag_counter.get(tag, 0) + 1
    print("Tag coverage:")
    for tag, count in sorted(tag_counter.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")

def top_entities(entities_json):
    """
    Reads entities_summary.json and returns a sorted list of (entity, freq) tuples.
    """
    if not os.path.exists(entities_json):
        print(f"Entities JSON not found: {entities_json}")
        return []
    with open(entities_json, "r") as f:
        data = json.load(f)
    return sorted(data.items(), key=lambda item: item[1], reverse=True)

def print_concepts_missing_examples(instr_dir):
    """
    Prints names of slides or concepts that are missing examples (if 'examples' is empty or missing).
    """
    missing = []
    files = [f for f in os.listdir(instr_dir) if f.endswith(".json")]
    for fname in files:
        with open(os.path.join(instr_dir, fname), "r") as f:
            data = json.load(f)
        # Assumes concepts are under 'concepts' and each has an 'examples' field
        for concept in data.get("concepts", []):
            if not concept.get("examples"):
                missing.append(concept.get("name", "(unnamed)"))
    if missing:
        print("Concepts missing examples:")
        for name in missing:
            print(f"  {name}")
    else:
        print("No concepts missing examples found.")

def visual_asset_report(instr_dir):
    """
    Returns a list of all image/visual assets referenced in instructional JSON files.
    """
    assets = []
    files = [f for f in os.listdir(instr_dir) if f.endswith(".json")]
    for fname in files:
        with open(os.path.join(instr_dir, fname), "r") as f:
            data = json.load(f)
        # Assumes images are under 'images' (list of filenames/paths)
        for img in data.get("images", []):
            assets.append(img)
    return assets

def find_tagged_caption_segments(caption_dir, tag):
    """
    Returns all caption segments (from *_semantic.json files) where the segment's 'tags' field includes 'tag'.
    """
    tagged = []
    for fname in os.listdir(caption_dir):
        if not fname.endswith("_semantic.json"):
            continue
        with open(os.path.join(caption_dir, fname), "r") as f:
            segments = json.load(f)
        for seg in segments:
            seg_tags = seg.get("tags", [])
            if tag in seg_tags:
                tagged.append(seg)
    return tagged

def assessments_by_topic(topics_summary_path):
    """
    Returns a dict of topic_id to count of segments, optionally only where segments have 'assessment' or similar tag.
    """
    if not os.path.exists(topics_summary_path):
        return {}
    with open(topics_summary_path, "r") as f:
        segments = json.load(f)
    topic_counts = {}
    for seg in segments:
        topic = seg.get("topic_id", "no_topic")
        # Optionally, only count if 'assessment' tag present
        if "assessment" in seg.get("tags", []):
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
    return topic_counts

def export_instructional_review(instr_dir, out_csv):
    """
    Writes a CSV listing each instructional file's title and a summary (if present).
    """
    files = [f for f in os.listdir(instr_dir) if f.endswith(".json")]
    header = ["filename", "title", "summary"]
    rows = []
    for fname in files:
        with open(os.path.join(instr_dir, fname), "r") as f:
            data = json.load(f)
        title = data.get("title", "")
        summary = data.get("summary", "")
        rows.append([fname, title, summary])
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Exported instructional review table to {out_csv}")

def export_needs_assessment(topics_summary_path, out_txt):
    """
    Writes a plain text list of concepts or segments needing assessment (e.g., those missing an 'assessment' tag).
    """
    needs = []
    if not os.path.exists(topics_summary_path):
        with open(out_txt, "w") as f:
            f.write("No topics summary available.\n")
        print(f"Exported concepts needing assessments to {out_txt}")
        return
    with open(topics_summary_path, "r") as f:
        segments = json.load(f)
    for seg in segments:
        # Add to list if 'assessment' not present in tags
        tags = seg.get("tags", [])
        if "assessment" not in tags:
            needs.append(seg.get("text", "[no text]"))
    with open(out_txt, "w") as f:
        for item in needs:
            f.write(f"{item}\n")
    print(f"Exported concepts needing assessments to {out_txt}")
