import logging
import datetime
from pathlib import Path
from course_compiler.config import OUTPUT_ROOT
from course_compiler.file_utils import sanitize_filename
from course_compiler.aggregate_and_enrich import aggregate_all

Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
logfile = Path(OUTPUT_ROOT) / f"run_aggregate_metadata_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    filename=logfile,
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO
)

def main():
    instructional_json_dir = Path(OUTPUT_ROOT) / "instructional_json"
    caption_prepped_dir = Path(OUTPUT_ROOT) / "caption_prepped"
    media_metadata_dir = Path(OUTPUT_ROOT) / "media_metadata"
    master_output = Path(OUTPUT_ROOT) / "master_summary.json"
    entities_output = Path(OUTPUT_ROOT) / "entities_summary.json"
    topics_output = Path(OUTPUT_ROOT) / "topics_summary.json"

    logging.info("Starting aggregation and enrichment.")
    try:
        result = aggregate_all(
            instructional_json_dir,
            caption_segments_dir=caption_prepped_dir,
            media_metadata_dir=media_metadata_dir,
            output_path=master_output,
            entities_output_path=entities_output,
            topics_output_path=topics_output,
            run_entity_extraction=True,
            run_topic_modeling=True,
            topic_model_params={"nr_topics": "auto"},
        )

        # Example: If you generate any per-topic/entity outputs, sanitize those filenames:
        # for topic in result.get("topics", []):
        #     safe_topic = sanitize_filename(topic["name"])
        #     out_path = Path(OUTPUT_ROOT) / f"topic_{safe_topic}.json"
        #     with open(out_path, "w") as f:
        #         json.dump(topic, f)

        logging.info(f"Aggregation complete. Master: {master_output}, Entities: {entities_output}, Topics: {topics_output}")
    except Exception as e:
        logging.error(f"Aggregation failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()

# This script aggregates instructional content, caption segments, and media metadata,
# performing entity extraction and topic modeling on the caption segments.
# It outputs a master summary JSON file and separate files for entities and topics.
# It ensures all necessary directories are specified and outputs the results in a structured format.
# It can be extended to include more metadata or processing steps as needed.
