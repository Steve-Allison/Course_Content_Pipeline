import os
import glob
import json
import argparse
from course_compiler.logging_utils import setup_logger
logger = setup_logger(__name__)
from sentence_transformers import SentenceTransformer, util

def load_segments(output_dir):
    files = glob.glob(os.path.join(output_dir, "*.json")) + glob.glob(os.path.join(output_dir, "captions", "*.json"))
    segments = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        base = os.path.basename(path)
        if isinstance(data, dict) and "segments" in data:
            for seg in data["segments"]:
                segments.append((seg["id"], seg.get("text", ""), base))
        elif isinstance(data, list):
            for seg in data:
                segments.append((seg["id"], seg.get("text", ""), base))
    return segments

def detect_duplicates(segments, model_name, threshold):
    model = SentenceTransformer(model_name)
    texts = [s[1] for s in segments]
    embeddings = model.encode(texts, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings)

    results = []
    seen = set()
    for i in range(len(segments)):
        for j in range(i+1, len(segments)):
            if sim_matrix[i][j] > threshold:
                pair = (segments[i][0], segments[j][0])
                if pair not in seen:
                    results.append({
                        "segment_a": segments[i][0],
                        "segment_b": segments[j][0],
                        "similarity": float(sim_matrix[i][j]),
                        "source_a": segments[i][2],
                        "source_b": segments[j][2]
                    })
                    seen.add(pair)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect near-duplicate segments across content.")
    parser.add_argument("--output_dir", default="output", help="Path to output folder")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--threshold", type=float, default=0.9, help="Similarity threshold (0-1)")
    parser.add_argument("--output_file", help="Optional output filename", default="duplicate_segments.json")
    args = parser.parse_args()

    segments = load_segments(args.output_dir)
    duplicates = detect_duplicates(segments, args.model, args.threshold)

    output_file = os.path.join(args.output_dir, args.output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(duplicates, f, indent=2)

    logger.info(f"Found {len(duplicates)} duplicate segment pairs.")
    logger.info(f"Saved to {output_file}")