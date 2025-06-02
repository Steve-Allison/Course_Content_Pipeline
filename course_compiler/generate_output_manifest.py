import os
import json
import argparse
from course_compiler.logging_utils import setup_logger

logger = setup_logger(__name__)

def load_manifest(manifest_path):
    if not os.path.exists(manifest_path):
        logger.warning("Manifest file not found: %s", manifest_path)
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt_bundle(manifest, output_dir, bundle_name):
    prompts = []

    glossary_ai_path = manifest.get("glossary", {}).get("ai_prompt_path")
    if glossary_ai_path and os.path.exists(glossary_ai_path):
        with open(glossary_ai_path, "r", encoding="utf-8") as f:
            prompts.append("### Glossary AI Prompts\n" + f.read())

    if not prompts:
        logger.warning("No prompt content found in manifest.")
        return

    bundle_path = os.path.join(output_dir, bundle_name)
    with open(bundle_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(prompts))

    logger.info("Prompt bundle written to %s", bundle_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an AI prompt bundle from output manifest.")
    parser.add_argument("--manifest", default="output/project_output_manifest.json", help="Path to manifest JSON")
    parser.add_argument("--output_dir", default="output", help="Directory to write prompt bundle")
    parser.add_argument("--bundle_name", default="prompt_bundle.md", help="Filename for bundled prompts")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    build_prompt_bundle(manifest, args.output_dir, args.bundle_name)