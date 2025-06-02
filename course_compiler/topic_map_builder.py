import json
import os
from typing import List, Dict
from content_schema import ContentItem
import re
from difflib import get_close_matches


def load_content_items(json_paths: List[str]) -> List[ContentItem]:
    items = []
    for path in json_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            items.append(ContentItem(
                source=data['source'],
                media_type=data['media_type'],
                segments=data['segments']
            ))
    return items


def normalize_topic(topic: str) -> str:
    topic = topic.lower()
    topic = re.sub(r'[^a-z0-9 ]', '', topic)
    topic = re.sub(r'\b(the|of|and|in|on|a|an|to|process|concept|understanding)\b', '', topic)
    return topic.strip()


def build_topic_map(content_items: List[ContentItem], fuzzy_merge: bool = False) -> Dict[str, List[Dict]]:
    topic_map = {}
    topic_index = {}  # for fuzzy topic normalization if enabled

    for item in content_items:
        for segment in item.segments:
            topics = segment.get('topics', [])
            for topic in topics:
                entry = {
                    "source": item.source,
                    "segment_id": segment['id'],
                    "summary": segment.get('summary', segment.get('text', '')[:100]),
                    "timestamp": segment.get('timestamp', None)
                }

                if fuzzy_merge:
                    norm = normalize_topic(topic)
                    match = get_close_matches(norm, topic_index.keys(), n=1, cutoff=0.85)
                    if match:
                        canonical = topic_index[match[0]]
                    else:
                        canonical = topic
                        topic_index[norm] = topic
                else:
                    canonical = topic

                topic_map.setdefault(canonical, []).append(entry)

    return topic_map


def save_topic_map(topic_map: Dict[str, List[Dict]], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(topic_map, f, indent=2, ensure_ascii=False)


# Example usage
if __name__ == '__main__':
    input_files = ['output/structured_content.json']  # or glob('output/*.json')
    output_topic_map = 'output/topic_map.json'

    items = load_content_items(input_files)
    topic_map = build_topic_map(items, fuzzy_merge=True)
    save_topic_map(topic_map, output_topic_map)
    print(f"Saved topic map with {len(topic_map)} topics to {output_topic_map}")
