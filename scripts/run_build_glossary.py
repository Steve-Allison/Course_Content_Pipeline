from course_compiler.config import OUTPUT_ROOT
from course_compiler.glossary_builder import build_glossary
from pathlib import Path

def main():
    instr_dir = Path(OUTPUT_ROOT) / "instructional_json"
    out_json = Path(OUTPUT_ROOT) / "glossary_for_definition.json"
    out_csv = Path(OUTPUT_ROOT) / "glossary_for_definition.csv"

    files = list(instr_dir.glob("*.json"))
    build_glossary(files, out_json, out_csv)
    print(f"Glossary ready for SME/LLM definition written to {out_json}")
    +    # Generate JSON glossary
+    build_glossary(files, out_json)
+    print(f"Glossary JSON written to {out_json}")
+
+    # Now convert JSON to CSV
+    import json, csv
+    with open(out_json, "r") as fj, open(out_csv, "w", newline="") as fc:
+        glossary = json.load(fj)
+        writer = csv.DictWriter(fc, fieldnames=["term", "definition", "example", "difficulty"])
+        writer.writeheader()
+        for entry in glossary:
+            writer.writerow(entry)
+    print(f"Glossary CSV written to {out_csv}")

if __name__ == "__main__":
    main()
