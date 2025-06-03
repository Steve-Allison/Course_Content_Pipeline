# Course_content_compiler

> **Modular, Mac-Optimized Python Pipeline for Instructional and Multimodal Content Analysis**

---

## **Project Overview**

This pipeline processes corporate learning and instructional content in bulk—including PowerPoint, Word, PDF, audio, video, and captions (VTT/SRT)—to generate rich, structured, and AI-ready datasets with **comprehensive quality analysis and optimization recommendations**.

## Supported File Types

- Microsoft Excel (.xlsx, .xlsm, .xltx, .xls)
- Microsoft PowerPoint (.pptx, .ppt)
- Microsoft Word (.docx, .doc)
- PDF (.pdf)
- Text files (.txt, .md, .rst, .csv, .json, .xml, .html)

- **Extracts instructional hierarchy, images, tables, charts, and speaker notes from docs**
- **Performs robust semantic analysis and tagging using NLP/LLMs**
- **Processes and aligns captions (VTT/SRT) with full instructional tagging**
- **Runs batch entity extraction and topic modeling**
- **Preps all audio/video, OCRs images, and logs every action**
- **Extracts detailed image metadata and chart data from PPTX**
- **Extracts tables from PDFs using Camelot**
- **🆕 Comprehensive content quality analysis and optimization**
- **🆕 Automated duplicate detection using AI embeddings**
- **🆕 Glossary usage optimization with AI-ready improvement prompts**
- **All outputs and logs are auto-organized under a dedicated `/output/` folder**

---

## **3-Phase Pipeline Architecture**

### **🔧 Phase 1: Content Preparation**

- Audio/Video file processing and metadata extraction
- Instructional content extraction (PPTX, DOCX, PDF)
- Caption processing and semantic tagging
- Media transcription (ASR for files without captions)

### **📊 Phase 2: Core Analysis & Enrichment**

- Content summarization and glossary building
- Entity extraction and topic modeling
- Meta-analysis and prompt generation
- Fuzzy topic mapping

### **🔍 Phase 3: Comprehensive Analysis** _(NEW!)_

- **Source distribution analysis** - Track content mix across formats
- **Quality assessment** - Identify content needing improvement
- **Glossary optimization** - Remove unused terms, enhance definitions
- **Duplicate detection** - Find redundant content using AI similarity
- **Output manifest** - Catalog all results with actionable insights

---

## **Directory Structure**

```plaintext
Course_content_compiler/
├── main.py                       # One-command pipeline runner (3 phases)
├── README.md
├── course_compiler/              # Modular code package
│   ├── aggregate_and_enrich.py
│   ├── audio_prep.py
│   ├── config.py
│   ├── extractor.py              # Extracts speaker notes, tables, charts, image metadata, file metadata
│   ├── extractors/               # New class-based extractor architecture
│   ├── file_utils.py
│   ├── metadata.py
│   ├── nlp_tagging.py
│   ├── ocr.py
│   ├── summarisation.py          # Flattens all nested instructional content for summary
│   ├── video_prep.py
│   ├── vtt_srt_prep.py
│   ├── analyze_segment_sources.py       # 🆕 Content source analysis
│   ├── analyze_segment_quality_and_topics.py  # 🆕 Quality metrics
│   ├── analyze_glossary_usage.py        # 🆕 Glossary optimization
│   ├── detect_duplicates.py             # 🆕 AI-powered duplicate detection
│   └── generate_output_manifest.py      # 🆕 Output cataloging
└── scripts/
    ├── prep_instructional_content.py
    ├── prep_caption_alignment.py
    ├── run_aggregate_metadata.py
    ├── run_media_metadata.py     # Extracts metadata from prepped media, not originals
    ├── run_comprehensive_analysis.py  # 🆕 Automated analysis suite
    └── (audio/video batch scripts, optional)
```

---

## **Input/Output Folder Convention**

- **Inputs:**
  Place all raw files in
  `/Users/steveallison/Downloads/Course_content_compiler/original_input/`

- **Outputs:**
  All results (logs, JSON, images, analysis reports) will be auto-organized under
  `/Users/steveallison/Downloads/Course_content_compiler/output/`

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
   # For advanced PDF table extraction:
   pip install "camelot-py[cv]" pdfplumber pillow
   ```

   _(See end for sample `requirements.txt`)_

3. **Verify your input files are in the correct folder!**

---

## **How to Run the Full Pipeline**

From the project root:

```sh
python main.py --max_workers 4
```

- This will run all **3 phases automatically**:

  **🔧 Phase 1: Content Preparation**

  1. Extract instructional structure, images, tables, charts, and speaker notes from all PPTX, DOCX, PDF
  2. Process and semantically tag all VTT/SRT caption files
  3. Extract media metadata and transcribe audio/video without captions

  **📊 Phase 2: Core Analysis & Enrichment** 4. Aggregate all results, run entity extraction and topic modeling 5. Generate glossaries, topic maps, and meta-analysis reports

  **🔍 Phase 3: Comprehensive Analysis** _(NEW!)_ 6. **Automated quality assessment and optimization recommendations**

---

## **Key Output Folders**

- `/output/instructional_json/`

  - Hierarchical JSON for each doc, plus all slide/page images (in `/images/`)
  - **Includes:**
    - Speaker notes (PPTX)
    - Tables (PPTX, DOCX, PDF)
    - Chart data (PPTX)
    - Image metadata (all formats)
    - File metadata (all formats)

- `/output/summary/`

  - **instructional_summary.jsonl / .csv:**
    - **Flattens all nested instructional content:** every slide, page, section, and component is a row
    - All extracted fields (text, tags, images, tables, charts, speaker notes, etc.)
  - **summary_glossary.json / .md:**
    - Deduplicated glossary with sample contexts

- `/output/media_metadata/`

  - **Contains metadata for prepped audio and video files** (from `audio_prepped` and `video_prepped`)

- `/output/` **🆕 Analysis Reports:**

  - **segment_quality_report.json** - Content quality metrics and warnings
  - **topic_coverage_report.json** - Topic distribution analysis
  - **segment_counts_by_source.json** - Content source breakdown
  - **duplicate_segments.json** - AI-detected similar content pairs
  - **glossary_term_usage.json** - Term frequency analysis
  - **unused_glossary_terms.json** - Terms to review/remove
  - **glossary_improvement_prompts.txt** - Human-readable recommendations
  - **glossary_ai_prompt_templates.txt** - AI-ready enhancement prompts

- `/output/*.log`
  - Timestamped logs for every batch script and analysis module

---

## **🆕 Comprehensive Analysis Features**

The pipeline now includes intelligent quality assessment and optimization:

### **Content Quality Analysis**

- **Segment Quality Metrics:** Identifies content that's too short, missing summaries/tags
- **Topic Coverage Analysis:** Shows topic distribution and gaps
- **Quality Warnings:** Flags issues exceeding configurable thresholds

### **Glossary Optimization**

- **Usage Tracking:** Counts how often each term appears in content
- **Unused Term Detection:** Identifies terms that aren't referenced
- **AI Enhancement Prompts:** Generates ready-to-use prompts for improving definitions

### **Duplicate Content Detection**

- **Semantic Similarity:** Uses sentence transformers to find similar content
- **Configurable Threshold:** Default 85% similarity for duplicate detection
- **Source Tracking:** Shows which documents contain similar content

### **Source Distribution Analysis**

- **Content Mix Tracking:** Counts segments by source (captions, slides, PDFs, etc.)
- **Portfolio Overview:** Understand content balance across formats

### **Output Cataloging**

- **Manifest Generation:** Creates comprehensive index of all pipeline outputs
- **Prompt Bundling:** Consolidates AI-ready improvement prompts

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
- **Parallel Enrichment with Thread Pooling**
  Accelerates segment processing using `--max_workers` to distribute tasks across threads.
- **Function-Level Disk Caching**
  Avoids redundant processing by caching results of enrichment functions like summarization and glossary extraction.

---

## **Pipeline Scripts**

**(Do not run these manually unless debugging—`main.py` orchestrates everything)**

### **Phase 1: Content Preparation**

- `scripts/prep_instructional_content.py`
  → Extracts docs, hierarchy, images, tables, charts, speaker notes, image metadata, file metadata
- `scripts/prep_caption_alignment.py`
  → Processes captions, applies full NLP/LLM tagging
- `scripts/run_media_metadata.py`
  → Extracts metadata from prepped audio/video files

### **Phase 2: Core Analysis & Enrichment**

- `scripts/run_aggregate_metadata.py`
  → Aggregates all outputs, runs entity extraction and topic modeling
- `scripts/summarisation_run.py`
  → Generates summaries and glossaries with flattening logic
- `scripts/enrichment_meta_run.py`
  → Meta-analysis and prompt/QA file generation

### **🆕 Phase 3: Comprehensive Analysis**

- `scripts/run_comprehensive_analysis.py`
  → **Automated suite running all 5 analysis modules:**
  - Source distribution analysis
  - Content quality and topic analysis
  - Glossary usage optimization
  - Duplicate content detection
  - Output manifest generation

### **Standalone Analysis Scripts** _(Run automatically in Phase 3)_

- `scripts/analyze_segment_sources.py`
  → Counts segments by content source (`caption`, `instructional`, etc.)
- `scripts/analyze_segment_quality_and_topics.py`
  → Reports segment issues and topic tagging coverage
- `scripts/analyze_glossary_usage.py`
  → Tracks glossary term usage and generates improvement prompts
- `scripts/detect_duplicates.py`
  → Finds semantically similar (duplicate) segments across content
- `scripts/generate_output_manifest.py`
  → Indexes all pipeline outputs into a single manifest

---

## **Business Value & Use Cases**

### **Content Portfolio Optimization**

- **Quality Control:** Automated identification of content needing improvement
- **Duplicate Elimination:** Find redundant materials for consolidation
- **Glossary Cleanup:** Remove unused terms to reduce cognitive load

### **Instructional Design Insights**

- **Topic Gap Analysis:** Identify under-covered topics
- **Content Balance:** Track mix of formats (video, slides, text)
- **Learning Path Optimization:** Use duplicate detection to create learning progressions

### **AI-Powered Content Enhancement**

- **Automated Improvement:** Generated prompts ready for ChatGPT/Claude
- **Scalable Reviews:** Process hundreds of documents with consistent quality metrics
- **Smart Recommendations:** Data-driven suggestions for content creators

---

## **Troubleshooting**

- **No output/logs?**
  - Double-check your input path in `.env` file and that files are present
  - Make sure you are running from the project root (`cd .../Course_content_compiler/`)
  - Check logs in `/output/` for any tracebacks
- **Import/module errors?**
  - Make sure `PYTHONPATH` is set in `main.py` subprocess calls (already configured)
  - Ensure your conda environment is activated: `conda activate course_pipeline`
- **Images, tables, or charts missing?**
  - Ensure you have installed all advanced dependencies (see Setup)
  - The extractor includes speaker notes, tables, charts, and image metadata for all supported formats
- **Analysis reports empty?**
  - Run the full pipeline first to generate content, then analysis reports will be populated
  - Check that Phase 1 and 2 completed successfully before Phase 3 analysis

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
camelot-py[cv]
pdfplumber
python-dotenv
whisper
```

- To enable caching, ensure `joblib` is installed.

---

## **VS Code Project Settings**

- The project includes a `.vscode/settings.json` file that:
  - Sets the Python interpreter to your `course_pipeline` environment
  - Enables linting, Prettier formatting, and consistent editor settings
  - Excludes build and cache files from search and explorer

---

## **Attribution & Support**

- **Developed by:** Steve Allison and ChatGPT (OpenAI)
- **Pipeline is modular, Mac-native, and ready for scale or AI-powered batch work**

**For help or bug fixes, paste logs or errors into ChatGPT or GitHub Issues for a rapid response!**

---

**You now have a robust, production-grade, fully modular learning content pipeline with comprehensive quality analysis and AI-powered optimization recommendations.
Happy compiling!**
