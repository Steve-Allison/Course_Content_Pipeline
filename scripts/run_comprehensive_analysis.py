import logging
import datetime
import sys
import subprocess
import os
from pathlib import Path
from course_compiler.config import ANALYSIS_DIR, GLOSSARY_DIR, PROMPTS_DIR, LOGS_DIR, SUMMARY_DIR

# Set up logging
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
logfile = Path(LOGS_DIR) / f"comprehensive_analysis_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
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

    # Get the course_compiler directory and scripts directory
    course_compiler_dir = Path(__file__).parent.parent / "course_compiler"
    scripts_dir = Path(__file__).parent

    # Define all analysis scripts with their configurations
    analysis_scripts = [
        {
            "script": course_compiler_dir / "analyze_segment_sources.py",
            "args": ["--output_dir", SUMMARY_DIR, "--output_file", "analysis/segment_counts_by_source.json"],
            "description": "Source Distribution Analysis"
        },
        {
            "script": course_compiler_dir / "analyze_segment_quality_and_topics.py",
            "args": [
                "--output_dir", SUMMARY_DIR,
                "--quality_file", "analysis/segment_quality_report.json",
                "--topic_file", "analysis/topic_coverage_report.json",
                "--warnings_file", "analysis/segment_quality_warnings.json"
            ],
            "description": "Content Quality & Topic Analysis"
        },
        {
            "script": course_compiler_dir / "analyze_glossary_usage.py",
            "args": [
                "--output_dir", SUMMARY_DIR,
                "--usage_file", "glossary/term_usage.json",
                "--unused_file", "glossary/unused_terms.json",
                "--improvement_file", "glossary/improvement_prompts.txt",
                "--ai_prompt_file", "glossary/ai_prompt_templates.txt"
            ],
            "description": "Glossary Usage Analysis"
        },
        {
            "script": course_compiler_dir / "detect_duplicates.py",
            "args": ["--output_dir", SUMMARY_DIR, "--threshold", "0.85", "--output_file", "analysis/duplicate_segments.json"],
            "description": "Duplicate Content Detection"
        },
        {
            "script": scripts_dir / "run_advanced_audio_analysis.py",
            "args": [],
            "description": "🎵 Advanced Audio Feature Analysis"
        },
        {
            "script": scripts_dir / "run_enhanced_knowledge_extraction.py",
            "args": [],
            "description": "🧠 Enhanced Knowledge Extraction (Visual + Knowledge Graph)"
        },
        {
            "script": course_compiler_dir / "generate_output_manifest.py",
            "args": ["--output_dir", SUMMARY_DIR, "--bundle_name", "prompts/prompt_bundle.md"],
            "description": "Output Manifest Generation"
        }
    ]

    results = []

    print(f"Running {len(analysis_scripts)} analysis modules...")
    print(f"Summary directory: {SUMMARY_DIR}")

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

    # List generated files in organized structure
    print(f"\n=== Generated Analysis Files ===")

    # Analysis files
    analysis_files = [
        "analysis/segment_counts_by_source.json",
        "analysis/segment_quality_report.json",
        "analysis/topic_coverage_report.json",
        "analysis/segment_quality_warnings.json",
        "analysis/duplicate_segments.json",
        "analysis/audio_features/audio_analysis_summary.json",  # Advanced audio analysis
        "analysis/visual_content/visual_insights_summary.json",  # NEW: Visual content analysis
        "analysis/knowledge_graph/knowledge_graph.json",  # NEW: Knowledge graph
        "analysis/enhanced_insights/enhanced_ai_insights.json"  # NEW: Enhanced AI insights
    ]

    # Glossary files
    glossary_files = [
        "glossary/term_usage.json",
        "glossary/unused_terms.json",
        "glossary/improvement_prompts.txt",
        "glossary/ai_prompt_templates.txt"
    ]

    # Prompt files
    prompt_files = [
        "prompts/prompt_bundle.md",
        "analysis/enhanced_insights/enhanced_ai_prompts.md"  # NEW: Enhanced AI prompts
    ]

    all_files = analysis_files + glossary_files + prompt_files

    for filename in all_files:
        filepath = Path(SUMMARY_DIR) / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  📄 {filename} ({size_kb:.1f} KB)")

    # Special mention for enhanced analysis directories
    enhanced_dirs = [
        ("analysis/audio_features", "🎵 Advanced audio analysis"),
        ("analysis/visual_content", "🔍 Visual content analysis"),
        ("analysis/knowledge_graph", "🧠 Knowledge graph analysis"),
        ("analysis/enhanced_insights", "🎯 Enhanced AI insights")
    ]

    for dir_path, description in enhanced_dirs:
        full_dir_path = Path(SUMMARY_DIR) / dir_path
        if full_dir_path.exists():
            file_count = len(list(full_dir_path.rglob("*.json"))) + len(list(full_dir_path.rglob("*.md")))
            if file_count > 0:
                print(f"  {description}: {file_count} analysis files")

    logging.info(f"Comprehensive analysis complete: {successful}/{total} successful")
    print(f"\nDetailed logs: {logfile}")
    print(f"🎯 All AI-ready files in: {SUMMARY_DIR}")
    print(f"🚀 Enhanced knowledge extraction now includes visual analysis and knowledge graphs!")

if __name__ == "__main__":
    main()
