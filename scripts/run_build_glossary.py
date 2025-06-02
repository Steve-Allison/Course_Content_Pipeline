from course_compiler.config import OUTPUT_ROOT
from course_compiler.glossary_builder import build_glossary
from pathlib import Path

def main():
    instr_dir = Path(OUTPUT_ROOT) / "instructional_json"
    out_json = Path(OUTPUT_ROOT) / "glossary_for_definition.json"
    out_csv = Path(OUTPUT_ROOT) / "glossary_for_definition.csv"

    # Create a simple input file from extracted content
    all_text = ""
    files = list(instr_dir.glob("*.json"))
    if files:
        import json
        for file in files:
            with open(file, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    # Extract text from the data structure
                    def extract_text(obj):
                        text = ""
                        if isinstance(obj, dict):
                            for v in obj.values():
                                text += extract_text(v) + " "
                        elif isinstance(obj, list):
                            for item in obj:
                                text += extract_text(item) + " "
                        elif isinstance(obj, str):
                            text += obj + " "
                        return text
                    all_text += extract_text(data)

    # Write all text to a temporary file for glossary building
    temp_input = Path(OUTPUT_ROOT) / "temp_text_input.txt"
    with open(temp_input, "w") as f:
        f.write(all_text)

    # Generate JSON glossary
    build_glossary(temp_input, out_json)
    print(f"Glossary JSON written to {out_json}")

    # Now convert JSON to CSV
    import json, csv
    with open(out_json, "r") as fj, open(out_csv, "w", newline="") as fc:
        glossary = json.load(fj)
        writer = csv.DictWriter(fc, fieldnames=["term", "definition", "example", "difficulty"])
        writer.writeheader()
        for entry in glossary:
            writer.writerow(entry)
    print(f"Glossary CSV written to {out_csv}")

if __name__ == "__main__":
    main()
