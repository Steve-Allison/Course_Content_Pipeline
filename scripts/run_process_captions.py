import os
import glob
from vtt_srt_prep import load_and_process_captions
from content_schema import ContentItem, Segment
import json

INPUT_DIR = "input"
OUTPUT_DIR = "output/captions"

def add_source_tag(metadata: dict, tag: str) -> dict:
    return {**metadata, "source_tag": tag}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    caption_files = glob.glob(os.path.join(INPUT_DIR, "*.vtt")) + glob.glob(os.path.join(INPUT_DIR, "*.srt"))

    if not caption_files:
        print("No VTT or SRT files found.")
        return

    all_segments = []

    for cap_path in caption_files:
        print(f"Processing {cap_path}...")
        segments = load_and_process_captions(cap_path, OUTPUT_DIR)
        base_name = os.path.basename(cap_path)
        for seg in segments:
            all_segments.append(Segment(
                id=seg['id'],
                text=seg['text'],
                summary=seg.get('summary'),
                topics=seg.get('topics', []),
                tags=seg.get('tags', []) + ['caption'],
                timestamp=seg.get('timestamp'),
                glossary_terms=seg.get('glossary_terms', []),
                metadata=add_source_tag(seg.get('metadata', {}), "caption")
            ))

    if all_segments:
        content = ContentItem(
            source="captions_combined",
            media_type="caption",
            segments=all_segments
        )
        output_path = os.path.join(OUTPUT_DIR, "captions_combined.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content.to_json())
        print(f"Unified caption content written to {output_path}")

if __name__ == "__main__":
    main()