import logging
import datetime
import sys
import subprocess
import os
from pathlib import Path
from course_compiler.config import OUTPUT_ROOT

# Set up logging
Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
logfile = Path(OUTPUT_ROOT) / f"comprehensive_analysis_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    filename=logfile,
    filemode="a",
    format="%(asctime)s %(levelname)s:%(message)s",
    level=logging.INFO
)

def run_analysis_script(script_path, script_args=None):
    """Run an analysis script and handle errors gracefully."""
    script_args = script_args or []
    try:
        cmd = [sys.executable, str(script_path)] + script_args
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info(f"SUCCESS: {script_path.name}")
        if result.stdout:
            logging.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"FAILED: {script_path.name} - {e}")
        if e.stderr:
            logging.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"EXCEPTION running {script_path.name}: {e}")
        return False

def main():
    print("=== Running Comprehensive Content Analysis ===")
    logging.info("Starting comprehensive analysis suite...")

    # Get the course_compiler directory
    course_compiler_dir = Path(__file__).parent.parent / "course_compiler"
    output_dir = OUTPUT_ROOT

    # Define all analysis scripts with their configurations
    analysis_scripts = [
        {
            "script": course_compiler_dir / "analyze_segment_sources.py",
            "args": ["--output_dir", output_dir],
            "description": "Source Distribution Analysis"
        },
        {
            "script": course_compiler_dir / "analyze_segment_quality_and_topics.py",
            "args": ["--output_dir", output_dir],
            "description": "Content Quality & Topic Analysis"
        },
        {
            "script": course_compiler_dir / "analyze_glossary_usage.py",
            "args": ["--output_dir", output_dir],
            "description": "Glossary Usage Analysis"
        },
        {
            "script": course_compiler_dir / "detect_duplicates.py",
            "args": ["--output_dir", output_dir, "--threshold", "0.85"],
            "description": "Duplicate Content Detection"
        },
        {
            "script": course_compiler_dir / "generate_output_manifest.py",
            "args": ["--output_dir", output_dir],
            "description": "Output Manifest Generation"
        }
    ]

    results = []

    print(f"Running {len(analysis_scripts)} analysis modules...")
    print(f"Output directory: {output_dir}")

    for i, config in enumerate(analysis_scripts, 1):
        script = config["script"]
        args = config["args"]
        desc = config["description"]

        print(f"\n[{i}/{len(analysis_scripts)}] {desc}")
        print(f"Script: {script.name}")

        if not script.exists():
            print(f"  ❌ SKIP: Script not found")
            logging.warning(f"Script not found: {script}")
            results.append((desc, False, "Script not found"))
            continue

        success = run_analysis_script(script, args)
        if success:
            print(f"  ✅ COMPLETED")
            results.append((desc, True, "Success"))
        else:
            print(f"  ❌ FAILED")
            results.append((desc, False, "Error during execution"))

    # Summary report
    print(f"\n=== Analysis Complete ===")
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"Results: {successful}/{total} analysis modules completed successfully")

    for desc, success, status in results:
        status_icon = "✅" if success else "❌"
        print(f"  {status_icon} {desc}: {status}")

    # List generated files
    print(f"\n=== Generated Analysis Files ===")
    analysis_files = [
        "segment_counts_by_source.json",
        "segment_quality_report.json",
        "topic_coverage_report.json",
        "segment_quality_warnings.json",
        "glossary_term_usage.json",
        "unused_glossary_terms.json",
        "glossary_improvement_prompts.txt",
        "glossary_ai_prompt_templates.txt",
        "duplicate_segments.json",
        "prompt_bundle.md"
    ]

    for filename in analysis_files:
        filepath = Path(output_dir) / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  📄 {filename} ({size_kb:.1f} KB)")

    logging.info(f"Comprehensive analysis complete: {successful}/{total} successful")
    print(f"\nDetailed logs: {logfile}")

if __name__ == "__main__":
    main()
