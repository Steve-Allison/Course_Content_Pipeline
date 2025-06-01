import json
from pathlib import Path
from collections import Counter

def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def aggregate_instructional_json(json_folder):
    """
    Loads all instructional JSONs and returns as a list.
    """
    json_files = list(Path(json_folder).glob("*.json"))
    all_data = []
    for jf in json_files:
        data = load_json(jf)
        if data:
            all_data.append(data)
    return all_data

def aggregate_caption_segments(caption_folder):
    """
    Loads all semantic caption JSONs (VTT/SRT) and returns as a single flat list of segments.
    """
    caption_files = list(Path(caption_folder).glob("*_semantic.json"))
    all_segments = []
    for f in caption_files:
        data = load_json(f)
        if data and isinstance(data, list):
            all_segments.extend(data)
    return all_segments

def aggregate_media_metadata(media_metadata_folder):
    """
    Loads all audio/video metadata JSONs and returns as a dict by filename.
    """
    meta_files = list(Path(media_metadata_folder).glob("*.json"))
    metadata = {}
    for f in meta_files:
        metadata[f.stem] = load_json(f)
    return metadata

def batch_entity_extraction(processed_segments):
    """
    Accepts a list of processed segments (each with 'entities' field).
    Returns a frequency dictionary of all entities found.
    """
    all_entities = []
    for seg in processed_segments:
        if "entities" in seg and seg["entities"]:
            all_entities.extend(seg["entities"])
    freq = Counter(all_entities)
    return dict(freq)

def batch_topic_modeling(processed_segments, nr_topics="auto"):
    """
    Accepts a list of segments with 'cleaned_text' or 'text'.
    Returns topic assignments and model for each segment.
    """
    from bertopic import BERTopic
    texts = [seg.get("cleaned_text", seg.get("text", "")) for seg in processed_segments]
    topic_model = BERTopic(nr_topics=nr_topics)
    topics, probs = topic_model.fit_transform(texts)
    # Add topics back to each segment
    for i, seg in enumerate(processed_segments):
        seg["topic_id"] = int(topics[i]) if topics[i] is not None else -1
        seg["topic_label"] = topic_model.get_topic(int(topics[i])) if topics[i] is not None else None
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
):
    """
    Aggregates instructional, captions, and media metadata.
    Runs batch entity extraction and topic modeling on caption segments.
    Outputs a master JSON and (optionally) separate entity/topic files.
    Ensures topics_output_path and entities_output_path are always written, even if empty.
    """
    result = {
        "instructional_assets": aggregate_instructional_json(instructional_json_dir)
    }

    # --- Aggregate and process caption segments ---
    all_segments = []
    if caption_segments_dir:
        all_segments = aggregate_caption_segments(caption_segments_dir)
        result["caption_segments"] = all_segments

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
            print("Running topic modeling (BERTopic) on all caption segments...")
            topic_model, segments_with_topics = batch_topic_modeling(
                all_segments,
                nr_topics=(topic_model_params.get("nr_topics") if topic_model_params else "auto")
            )
            result["caption_segments"] = segments_with_topics
            segments_to_write = segments_with_topics
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate instructional JSON, captions, and media metadata."
    )
    parser.add_argument("--instr_dir", required=True, help="Folder of instructional JSON files")
    parser.add_argument("--captions_dir", help="Folder of caption-segment JSON files")
    parser.add_argument("--media_meta_dir", help="Folder of media metadata JSON files")
    parser.add_argument("--out_master", help="Path to write master JSON output")
    parser.add_argument("--out_entities", help="Path to write entities summary JSON")
    parser.add_argument("--out_topics", help="Path to write topics summary JSON")
    args = parser.parse_args()

    aggregate_all(
        instructional_json_dir=args.instr_dir,
        caption_segments_dir=args.captions_dir,
        media_metadata_dir=args.media_meta_dir,
        output_path=args.out_master,
        entities_output_path=args.out_entities,
        topics_output_path=args.out_topics,
    )
