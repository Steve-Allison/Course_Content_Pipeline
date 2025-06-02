

import os
import glob
from topic_map_builder import load_content_items, build_topic_map, save_topic_map

OUTPUT_ROOT = "output"
STRUCTURED_CONTENT_GLOB = os.path.join(OUTPUT_ROOT, "*.json")
TOPIC_MAP_PATH = os.path.join(OUTPUT_ROOT, "topic_map.json")

def main():
    content_files = glob.glob(STRUCTURED_CONTENT_GLOB)
    if not content_files:
        print("No content files found to process.")
        return

    items = load_content_items(content_files)
    topic_map = build_topic_map(items, fuzzy_merge=True)
    save_topic_map(topic_map, TOPIC_MAP_PATH)
    print(f"Saved topic map with {len(topic_map)} topics to {TOPIC_MAP_PATH}")

if __name__ == "__main__":
    main()