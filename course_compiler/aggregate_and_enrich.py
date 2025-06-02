import argparse
from course_compiler.logging_utils import setup_logger
logger = setup_logger(__name__)
import json
from pathlib import Path
from collections import Counter
from course_compiler.segment_enrichment import enrich_segments, Segment

# Helper: Add a source tag to metadata
def add_source_tag(metadata: dict, tag: str) -> dict:
    return {**metadata, "source_tag": tag}

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.info(f"Error loading {path}: {e}")
        return None

def aggregate_instructional_json(json_folder, media_type="instructional"):
    """
    Loads all instructional JSONs and returns as a list.
    Adds a source tag to each segment's metadata.
    """
    json_files = list(Path(json_folder).glob("*.json"))
    all_data = []
    for jf in json_files:
        data = load_json(jf)
        if data:
            # If data is a dict with 'segments', tag each segment's metadata
            if isinstance(data, dict) and "segments" in data and isinstance(data["segments"], list):
                for seg in data["segments"]:
                    seg["metadata"] = add_source_tag(seg.get("metadata", {}), media_type)
            # If data is a list of segments, tag each segment's metadata
            elif isinstance(data, list):
                for seg in data:
                    if isinstance(seg, dict):
                        seg["metadata"] = add_source_tag(seg.get("metadata", {}), media_type)
            all_data.append(data)
    return all_data

def aggregate_caption_segments(caption_folder, media_type="caption"):
    """
    Loads all semantic caption JSONs (VTT/SRT) and returns as a single flat list of segments.
    Adds a source tag to each segment's metadata.
    """
    caption_files = list(Path(caption_folder).glob("*_semantic.json"))
    all_segments = []
    for f in caption_files:
        data = load_json(f)
        if data and isinstance(data, list):
            for seg in data:
                if isinstance(seg, dict):
                    seg["metadata"] = add_source_tag(seg.get("metadata", {}), media_type)
            all_segments.extend(data)
    return all_segments

def aggregate_media_metadata(media_metadata_folder):
    """
    Loads all audio/video metadata JSONs and returns as a dict by filename.
    Tags each metadata dict with "source_tag": "media".
    """
    meta_files = list(Path(media_metadata_folder).glob("*.json"))
    metadata = {}
    for f in meta_files:
        meta = load_json(f)
        if isinstance(meta, dict):
            meta = add_source_tag(meta, "media")
        metadata[f.stem] = meta
    return metadata

def batch_entity_extraction(processed_segments):
    """
    Accepts a list of processed segments (each with 'entities' field).
    Returns a frequency dictionary of all entities found.
    Applies source tagging to segment metadata.
    """
    all_entities = []
    for seg in processed_segments:
        # Add source tag to segment metadata
        seg["metadata"] = add_source_tag(seg.get("metadata", {}), "caption")
        if "entities" in seg and seg["entities"]:
            all_entities.extend(seg["entities"])
    freq = Counter(all_entities)
    return dict(freq)

def batch_topic_modeling(processed_segments, nr_topics="auto"):
    """
    Accepts a list of segments with 'cleaned_text' or 'text'.
    Returns topic assignments and model for each segment.
    Applies source tagging to segment metadata.
    """
    from bertopic import BERTopic
    texts = [seg.get("cleaned_text", seg.get("text", "")) for seg in processed_segments]
    topic_model = BERTopic(nr_topics=nr_topics)
    topics, probs = topic_model.fit_transform(texts)
    # Add topics back to each segment
    for i, seg in enumerate(processed_segments):
        seg["topic_id"] = int(topics[i]) if topics[i] is not None else -1
        # If topic exists, get top words as label
        seg["topic_label"] = topic_model.get_topic(int(topics[i])) if topics[i] is not None else None
        # Add source tag to segment metadata
        seg["metadata"] = add_source_tag(seg.get("metadata", {}), "caption")
    return topic_model, processed_segments

def aggregate_all(
    instructional_json_dir,
    caption_segments_dir=None,
    media_metadata_dir=None,
    output_path=None,
    entities_output_path=None,
    topics_output_path=None,
    run_entity_extraction=True,
    run_topic_modeling=True,
    topic_model_params=None,
    max_workers=None,
):
    """
    Aggregates instructional, captions, and media metadata.
    Runs batch entity extraction and topic modeling on caption segments.
    Outputs a master JSON and (optionally) separate entity/topic files.
    Ensures topics_output_path and entities_output_path are always written, even if empty.
    """
    result = {
        "instructional_assets": aggregate_instructional_json(instructional_json_dir, media_type="instructional")
    }

    # --- Aggregate and process caption segments ---
    all_segments = []
    if caption_segments_dir:
        all_segments = aggregate_caption_segments(caption_segments_dir, media_type="caption")
        # Enrich segments with optional max_workers
        enriched_segments = enrich_segments(all_segments, max_workers=max_workers)
        result["caption_segments"] = enriched_segments
        all_segments = enriched_segments

    # --- Aggregate media metadata ---
    if media_metadata_dir:
        result["media_metadata"] = aggregate_media_metadata(media_metadata_dir)

    # --- Batch Entity Extraction ---
    entities_freq = {}
    if run_entity_extraction:
        if all_segments:
            entities_freq = batch_entity_extraction(all_segments)
        result["entity_summary"] = entities_freq
        if entities_output_path:
            with open(entities_output_path, "w") as fout:
                json.dump(entities_freq, fout, indent=2)

    # --- Batch Topic Modeling ---
    wrote_topics = False
    segments_to_write = []
    if run_topic_modeling:
        if all_segments:
            logger.info("Running topic modeling (BERTopic) on all caption segments...")
            topic_model, segments_with_topics = batch_topic_modeling(
                all_segments,
                nr_topics=(topic_model_params.get("nr_topics") if topic_model_params else "auto")
            )
            # Replace segments in result with topic-labeled segments
            result["caption_segments"] = segments_with_topics
            segments_to_write = segments_with_topics
        # Always write topics_output_path (even if empty)
        if topics_output_path:
            with open(topics_output_path, "w") as fout:
                json.dump(segments_to_write, fout, indent=2)
            wrote_topics = True

    # If topic modeling is off, still guarantee topics_output_path file exists (as empty list)
    if topics_output_path and not wrote_topics:
        with open(topics_output_path, "w") as fout:
            json.dump([], fout, indent=2)

    # --- Output master summary ---
    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate and enrich content segments.")
    parser.add_argument("--instructional_dir", required=True, help="Directory with instructional JSON files.")
    parser.add_argument("--caption_dir", help="Directory with caption JSON files.")
    parser.add_argument("--media_metadata_dir", help="Directory with media metadata JSON files.")
    parser.add_argument("--output_path", required=True, help="Path to write combined output JSON.")
    parser.add_argument("--entities_output_path", help="Optional path to write entity frequency JSON.")
    parser.add_argument("--topics_output_path", help="Optional path to write topic modeling JSON.")
    parser.add_argument("--disable_entities", action="store_true", help="Disable entity extraction step.")
    parser.add_argument("--disable_topics", action="store_true", help="Disable topic modeling step.")
    parser.add_argument("--max_workers", type=int, help="Number of threads for parallel enrichment")
    args = parser.parse_args()

    aggregate_all(
        instructional_json_dir=args.instructional_dir,
        caption_segments_dir=args.caption_dir,
        media_metadata_dir=args.media_metadata_dir,
        output_path=args.output_path,
        entities_output_path=args.entities_output_path,
        topics_output_path=args.topics_output_path,
        run_entity_extraction=not args.disable_entities,
        run_topic_modeling=not args.disable_topics,
        max_workers=args.max_workers
    )
