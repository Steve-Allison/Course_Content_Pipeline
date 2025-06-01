from course_compiler.config import OUTPUT_ROOT
+from course_compiler.enrichment_meta import (
+    group_segments_by_topic,
+    print_tag_coverage,
+    top_entities,
+    print_concepts_missing_examples,
+    visual_asset_report,
+    find_tagged_caption_segments,
+    assessments_by_topic,
+    export_instructional_review,
+    export_needs_assessment,
+)
from course_compiler.file_utils import sanitize_filename
import os
import sys

def file_exists_or_exit(path, description):
    """Check if a file exists, or print error and exit."""
    if not os.path.isfile(path):
        print(f"ERROR: Required file not found: {path}\n"
              f"Missing: {description}\n"
              f"Please run earlier pipeline steps and ensure input data is present.\n"
              f"Pipeline halted.")
        sys.exit(1)

def dir_exists_or_exit(path, description):
    """Check if a directory exists and is not empty, or print error and exit."""
    if not os.path.isdir(path) or len(os.listdir(path)) == 0:
        print(f"ERROR: Required directory missing or empty: {path}\n"
              f"Missing: {description}\n"
              f"Please ensure input files have been processed.\n"
              f"Pipeline halted.")
        sys.exit(1)

def main():
    instr_dir = f"{OUTPUT_ROOT}/instructional_json"
    caption_dir = f"{OUTPUT_ROOT}/caption_prepped"
    topics_summary = f"{OUTPUT_ROOT}/topics_summary.json"
    entities_json = f"{OUTPUT_ROOT}/entities_summary.json"
    outdir = f"{OUTPUT_ROOT}/prompt_topics"
    out_csv = f"{OUTPUT_ROOT}/instructional_review.csv"
    out_txt = f"{OUTPUT_ROOT}/concepts_needing_assessment.txt"

    # Pre-flight checks for all required files and directories
    file_exists_or_exit(topics_summary, "Topics summary JSON (topics_summary.json)")
    file_exists_or_exit(entities_json, "Entities summary JSON (entities_summary.json)")
    dir_exists_or_exit(instr_dir, "Instructional JSON directory (instructional_json)")
    dir_exists_or_exit(caption_dir, "Caption prepped directory (caption_prepped)")

    # 1. Group by topic and make prompt files
    # Here is where sanitize_filename should be used for any dynamic topic filenames
    group_segments_by_topic(topics_summary, outdir, filename_sanitizer=sanitize_filename)
    print(f"Saved topic prompt files to {outdir}")

    # 2. Print tag coverage
    print_tag_coverage(instr_dir)

    # 3. Top entities
    print("Top entities:")
    for entity, freq in top_entities(entities_json):
        # If writing per-entity output, use sanitize_filename(entity)
        print(f"{entity}: {freq}")

    # 4. Concepts missing examples
    print("Slides missing examples:")
    print_concepts_missing_examples(instr_dir)

    # 5. Visual asset report
    assets = visual_asset_report(instr_dir)
    print(f"Found {len(assets)} images in instructional content.")

    # 6. Find all caption "question" segments
    questions = find_tagged_caption_segments(caption_dir, "question")
    print(f"Found {len(questions)} question segments in captions.")

    # 7. Assessments by topic
    assessment_counts = assessments_by_topic(topics_summary)
    print("Assessments per topic:")
    for topic, count in assessment_counts.items():
        # If writing per-topic output, use sanitize_filename(topic)
        print(f"Topic {topic}: {count}")

    # 8. Export review CSV (name is fixed)
    export_instructional_review(instr_dir, out_csv)
    print(f"Exported instructional review table to {out_csv}")

    # 9. Export concepts needing assessments (name is fixed)
    export_needs_assessment(topics_summary, out_txt)
    print(f"Exported concepts needing assessments to {out_txt}")

if __name__ == "__main__":
    main()

# This script now supports full cross-platform safe output naming by passing sanitize_filename
# to any routine that generates per-topic, per-entity, or other dynamic output files.
# Ensure downstream functions like group_segments_by_topic accept a filename_sanitizer argument,
# or apply sanitize_filename whenever filenames are constructed from user/data input.
