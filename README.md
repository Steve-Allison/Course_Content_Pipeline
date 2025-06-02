# Course_content_compiler

> **Modular, Mac-Optimized Python Pipeline for Instructional and Multimodal Content Analysis**

---

## **Project Overview**

This pipeline processes corporate learning and instructional content in bulk—including PowerPoint, Word, PDF, audio, video, and captions (VTT/SRT)—to generate rich, structured, and AI-ready datasets.

* **Extracts instructional hierarchy and images from docs**
* **Performs robust semantic analysis and tagging using NLP/LLMs**
* **Processes and aligns captions (VTT/SRT) with full instructional tagging**
* **Runs batch entity extraction and topic modeling**
* **Preps all audio/video, OCRs images, and logs every action**
* **All outputs and logs are auto-organized under a dedicated `/output/` folder**

---

## **Directory Structure**

```plaintext
Course_content_compiler/
├── main.py                       # One-command pipeline runner
├── README.md
├── course_compiler/              # Modular code package
│   ├── aggregate_and_enrich.py
│   ├── audio_prep.py
│   ├── config.py
│   ├── extractor.py
│   ├── file_utils.py
│   ├── metadata.py
│   ├── nlp_tagging.py
│   ├── ocr.py
│   ├── summarisation.py
│   ├── video_prep.py
│   └── vtt_srt_prep.py
└── scripts/
    ├── prep_instructional_content.py
    ├── prep_caption_alignment.py
    ├── run_aggregate_metadata.py
    └── (audio/video batch scripts, optional)
```

---

## **Input/Output Folder Convention**

* **Inputs:**
  Place all raw files in
  `/Users/steveallison/Downloads/Course_content_complier/original_input/`

* **Outputs:**
  All results (logs, JSON, images, etc.) will be auto-organized under
  `/Users/steveallison/Downloads/Course_content_complier/output/`

---

## **Setup**

1. **Activate your environment**

   ```sh
   conda activate course_pipeline
   ```

2. **Install all requirements**

   ```sh
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   brew install ffmpeg tesseract
   ```

   *(See end for sample `requirements.txt`)*

3. **Verify your input files are in the correct folder!**

---

## **How to Run the Full Pipeline**

From the project root:

```sh
python main.py --max_workers 4
```

* This will:

  1. Extract instructional structure/images from all PPTX, DOCX, PDF
  2. Process and semantically tag all VTT/SRT caption files
  3. Aggregate all results, run entity extraction and topic modeling
  4. Output logs, JSONs, and images to `/output/` subfolders

---

## **Key Output Folders**

* `/output/instructional_json/`
  Hierarchical JSON for each doc, plus all slide/page images (in `/images/`)

* `/output/caption_prepped/`
  Semantic JSON for each VTT/SRT file (with instructional tags, entities, etc.)

* `/output/`

  * `master_summary.json`
    *(full aggregate summary)*
  * `entities_summary.json`
    *(entity frequency table)*
  * `topics_summary.json`
    *(topic modeling results)*

* `/output/*.log`
  Timestamped logs for every batch script

---

## **Advanced Analysis Features**

This pipeline also includes intelligent analysis layers that go beyond metadata extraction:

- **Segment Quality Checks**  
  Flags segments that are too short or missing summary/tags.

- **Topic Coverage & Frequency**  
  Reports how thoroughly topics are represented and which segments lack tagging.

- **Glossary Coverage & Usage**  
  Tracks which glossary terms are used, unused, and suggests terms for removal or improvement.

- **Duplicate Segment Detection**  
  Uses semantic similarity (Sentence Transformers) to find and log highly similar segment pairs.

- **AI Prompt Templates for Glossary Review**  
  Generates ready-to-use prompt files to feed AI tools for improving glossary definitions or examples.

- **Project Output Manifest**  
  A top-level JSON that indexes all output artifacts for use in dashboards or LLM agents.

  ```json
  {
    "glossary": {
      "terms_path": "...",
      "usage_path": "...",
      ...
    },
    "topics": { ... },
    "segments": { ... }
  }
  ```

- **Parallel Enrichment with Thread Pooling**  
  Accelerates segment processing using `--max_workers` to distribute tasks across threads.

- **Function-Level Disk Caching**  
  Avoids redundant processing by caching results of enrichment functions like summarization and glossary extraction.

---

## **Pipeline Scripts**

**(Do not run these manually unless debugging—`main.py` orchestrates everything)**

* `scripts/prep_instructional_content.py`
  → Extracts docs, hierarchy, images (logs to `/output/`)

* `scripts/prep_caption_alignment.py`
  → Processes captions, applies full NLP/LLM tagging (logs to `/output/`)

* `scripts/run_aggregate_metadata.py`
  → Aggregates all outputs, runs entity extraction and topic modeling (logs to `/output/`)

* `scripts/analyze_segment_sources.py`  
  → Counts segments by content source (`caption`, `instructional`, etc.)

* `scripts/analyze_segment_quality_and_topics.py`  
  → Reports segment issues and topic tagging coverage

* `scripts/analyze_glossary_usage.py`  
  → Tracks glossary term usage and generates improvement prompts

* `scripts/generate_output_manifest.py`  
  → Indexes all pipeline outputs into a single manifest

* `scripts/detect_duplicates.py`  
  → Finds semantically similar (duplicate) segments across content

---

## **Troubleshooting**

* **No output/logs?**

  * Double-check your input path in `course_compiler/config.py` and that files are present
  * Make sure you are running from the project root (`cd .../Course_content_compiler/`)
  * Check logs in `/output/` for any tracebacks

* **Import/module errors?**

  * Make sure `PYTHONPATH` is set in `main.py` subprocess calls (already patched above)
  * Do **not** add `__init__.py` files to `scripts/` (keep it for modular code only)

* **Images in wrong folder?**

  * The latest extractor always puts them in `/output/instructional_json/images/`

---

## **Sample requirements.txt**

```
spacy
transformers
torch
bertopic
scikit-learn
sentence-transformers
umap-learn
hdbscan
pytesseract
pillow
python-srt
webvtt-py
pymupdf
python-docx
python-pptx
```

* To enable caching, ensure `joblib` is installed.

---

## **Attribution & Support**

* **Developed by:** Steve Allison and ChatGPT (OpenAI)
* **Pipeline is modular, Mac-native, and ready for scale or AI-powered batch work**

**For help or bug fixes, paste logs or errors into ChatGPT or GitHub Issues for a rapid response!**

---

**You now have a robust, production-grade, fully modular learning content pipeline.
Happy compiling!**
