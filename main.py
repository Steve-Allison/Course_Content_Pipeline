import subprocess
import sys
import logging
import datetime
from pathlib import Path
import os
from course_compiler.config import OUTPUT_ROOT

Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
logfile = Path(OUTPUT_ROOT) / f"pipeline_main_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    filename=logfile,
    filemode="a",
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)

SCRIPTS_DIR = Path(__file__).parent / "scripts"

def run_script(script_path, desc):
    print(f"\n=== About to run: {script_path} ({desc}) ===")
    logging.info(f"=== STARTING: {desc} ({script_path}) ===")
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        logging.error(f"Script not found: {script_path}")
        sys.exit(1)
    try:
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent)}
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, check=True,
            env=env
        )
        print(result.stdout)
        logging.info(f"SUCCESS: {desc}\n{result.stdout}")
        if result.stderr:
            print(f"STDERR from {desc}:\n{result.stderr}")
            logging.warning(f"STDERR from {desc}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {desc}:\n{e.stderr or e}")
        logging.error(f"FAILED: {desc}\n{e.stderr or e}", exc_info=True)
        sys.exit(1)

def check_exists(path, desc, require_nonempty=False):
    p = Path(path)
    if not p.exists():
        print(f"\nERROR: Expected output missing: {path}\nDescription: {desc}\nCheck previous log/output and input data. Pipeline halted.")
        logging.error(f"Missing output: {path} ({desc})")
        sys.exit(1)
    if require_nonempty:
        if p.is_dir() and not any(p.iterdir()):
            print(f"\nERROR: Output directory is empty: {path}\nDescription: {desc}\nCheck prep phase for missing or invalid input. Pipeline halted.")
            logging.error(f"Empty output dir: {path} ({desc})")
            sys.exit(1)

def main():
    # Phase 1: PREP
    prep_steps = [
        {"script": SCRIPTS_DIR / "run_audio_prep.py", "desc": "Audio File Preparation"},
        {"script": SCRIPTS_DIR / "run_video_prep.py", "desc": "Video File Preparation"},
        {"script": SCRIPTS_DIR / "run_media_metadata.py", "desc": "Extract Media Metadata"},
        {"script": SCRIPTS_DIR / "run_transcribe_media.py", "desc": "Audio/Video Transcription (ASR for media without captions)"},
        {"script": SCRIPTS_DIR / "prep_instructional_content.py", "desc": "Instructional Content Extraction"},
        {"script": SCRIPTS_DIR / "prep_caption_alignment.py", "desc": "Caption (VTT/SRT & Transcript) Semantic Processing"},
    ]

    # Phase 2: ANALYSIS
    analysis_steps = [
        {"script": SCRIPTS_DIR / "summarisation_run.py", "desc": "Glossary & Coverage Summarisation", "outputs": []},
        {"script": SCRIPTS_DIR / "run_aggregate_metadata.py", "desc": "Aggregate and Enrich Outputs", "outputs": [
            {"path": f"{OUTPUT_ROOT}/entities_summary.json", "desc": "Entities summary JSON", "require_nonempty": False},
            {"path": f"{OUTPUT_ROOT}/topics_summary.json", "desc": "Topics summary JSON", "require_nonempty": False}
        ]},
        {"script": SCRIPTS_DIR / "enrichment_meta_run.py", "desc": "Meta-analysis and Prompt/QA File Generation", "outputs": []},
        {"script": SCRIPTS_DIR / "run_build_glossary.py", "desc": "Build Glossary for Definitions (SME/LLM)", "outputs": []},
        {"script": SCRIPTS_DIR / "run_build_topic_map.py", "desc": "Build Fuzzy Topic Map", "outputs": [
            {"path": f"{OUTPUT_ROOT}/topic_map.json", "desc": "Fuzzy Topic Map JSON", "require_nonempty": False}
        ]}
    ]

    start_time = datetime.datetime.now()
    print(f"\n=== Course_content_compiler Pipeline START: {start_time:%Y-%m-%d %H:%M:%S} ===")
    logging.info(f"Pipeline started at {start_time:%Y-%m-%d %H:%M:%S}")

    # Run all prep steps first, regardless of interim outputs
    for step in prep_steps:
        run_script(step["script"], step["desc"])

    # Now verify prep outputs before any analysis
    check_exists(f"{OUTPUT_ROOT}/media_metadata", "Media metadata directory", require_nonempty=True)
    check_exists(f"{OUTPUT_ROOT}/instructional_json", "Instructional JSON output", require_nonempty=True)
    check_exists(f"{OUTPUT_ROOT}/caption_prepped", "Caption-prepped directory", require_nonempty=True)

    # Now, run analysis/QA/aggregation phase only if prep successful
    for step in analysis_steps:
        run_script(step["script"], step["desc"])
        for output in step.get("outputs", []):
            check_exists(output["path"], output["desc"], output.get("require_nonempty", False))

    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n=== Pipeline COMPLETE! ({duration:.1f} seconds) ===")
    print(f"Check logs at: {logfile}")
    logging.info(f"Pipeline completed at {end_time:%Y-%m-%d %H:%M:%S} (duration: {duration:.1f} seconds)")

if __name__ == "__main__":
    main()
