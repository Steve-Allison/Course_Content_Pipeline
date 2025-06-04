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
- **🆕 Organized output structure with dedicated AI-ready files in `/summary/`**
- **🎵 NEW! Advanced audio analysis using librosa, torchaudio, Pydub & enhanced Whisper**

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

### **🔍 Phase 3: Comprehensive Analysis** _(ENHANCED!)_

- **Source distribution analysis** - Track content mix across formats
- **Quality assessment** - Identify content needing improvement
- **Glossary optimization** - Remove unused terms, enhance definitions
- **Duplicate detection** - Find redundant content using AI similarity
- **🎵 Advanced Audio Analysis** - Comprehensive audio feature extraction & quality assessment
- **Output manifest** - Catalog all results with actionable insights

---

## **🎯 Organized Output Structure**

The pipeline now creates a clean, organized output directory structure with all AI-ready files consolidated in the `summary/` folder:

```plaintext
output/
├── logs/                           # All pipeline logs (timestamped)
│   ├── pipeline_main_YYYYMMDD_HHMMSS.log
│   ├── prep_instructional_content_YYYYMMDD_HHMMSS.log
│   ├── advanced_audio_analysis_YYYYMMDD_HHMMSS.log  # 🎵 NEW!
│   └── ...
├── processed/                      # Raw processed content
│   ├── instructional_json/         # Extracted document content
│   │   ├── images/                 # All extracted images
│   │   └── *.json                  # Individual document extractions
│   ├── caption_prepped/            # Processed caption files
│   ├── audio_prepped/              # Processed audio files
│   ├── video_prepped/              # Processed video files
│   └── media_metadata/             # Media metadata files
├── summary/                        # 🎯 AI-READY FILES (Main target folder)
│   ├── instructional_summary.jsonl # Flattened content for AI
│   ├── instructional_summary.csv   # Flattened content for spreadsheet
│   ├── master_summary.json         # Aggregated all content
│   ├── entities_summary.json       # Named entities extracted
│   ├── topics_summary.json         # Topic modeling results
│   ├── manifest.json               # Complete output catalog
│   ├── glossary/                   # All glossary-related files
│   │   ├── summary_glossary.json   # Main glossary
│   │   ├── summary_glossary.md     # Human-readable glossary
│   │   ├── glossary_for_definition.json
│   │   ├── glossary_for_definition.csv
│   │   ├── term_usage.json         # Term frequency analysis
│   │   ├── unused_terms.json       # Terms to review/remove
│   │   ├── improvement_prompts.txt # Human-readable recommendations
│   │   └── ai_prompt_templates.txt # AI-ready enhancement prompts
│   ├── analysis/                   # Quality and content analysis
│   │   ├── segment_quality_report.json
│   │   ├── topic_coverage_report.json
│   │   ├── segment_quality_warnings.json
│   │   ├── duplicate_segments.json
│   │   ├── segment_counts_by_source.json
│   │   ├── topic_map.json
│   │   └── audio_features/         # 🎵 ADVANCED AUDIO ANALYSIS
│   │       ├── audio_analysis_summary.json    # Overall audio insights
│   │       ├── audio_prepped/      # Individual file analyses
│   │       │   └── *.json          # Per-file audio feature analysis
│   │       └── video_prepped/      # Video audio analyses
│   │           └── *.json          # Per-file video audio analysis
│   └── prompts/                    # AI prompt files and topic groupings
│       ├── topic_*.json            # Individual topic files
│       ├── instructional_review.csv
│       ├── concepts_needing_assessment.txt
│       └── prompt_bundle.md        # Consolidated AI prompts
└── temp/                           # Temporary files (can be deleted)
    └── temp_text_input.txt
```

---

## **🎵 Advanced Audio Analysis Features**

The pipeline now includes **comprehensive audio analysis** using industry-leading libraries:

### **🔬 Audio Quality Assessment**

- **Signal-to-Noise Ratio (SNR)** estimation for recording quality
- **Dynamic range** and **RMS energy** analysis
- **Spectral quality metrics** (brightness, rolloff, zero-crossing rate)
- **Loudness analysis** using multiple measurement standards

### **🎤 Enhanced Transcription & Speech Analysis**

- **Word-level timestamps** with confidence scores using enhanced Whisper
- **Language detection** and **speaking rate analysis** (WPM)
- **Voice activity detection** and **silence analysis**
- **Speech-to-silence ratio** and **pacing consistency**

### **🎭 Engagement & Accessibility Analysis**

- **Pitch variation** analysis (monotone vs. dynamic delivery)
- **Volume dynamics** and **speaking rhythm** assessment
- **Accessibility metrics**: clarity, frequency range, speaking rate appropriateness
- **Engagement indicators**: spectral flux, onset density, vocal variety

### **👥 Speaker Analysis & Content Structure**

- **Voice characteristics** using MFCC analysis
- **Speaker consistency** detection (single vs. multiple speakers)
- **Content structure boundaries** using onset and novelty detection
- **Tempo changes** indicating section transitions

### **🔍 Audio Fingerprinting & Duplicate Detection**

- **Chromagram and MFCC-based fingerprints** for duplicate detection
- **Spectral feature hashing** for efficient similarity comparison
- **Audio content similarity** analysis across the entire corpus

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
│   ├── advanced_audio_analysis.py       # 🎵 NEW! Comprehensive audio analysis
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
    ├── run_advanced_audio_analysis.py  # 🎵 NEW! Advanced audio feature analysis
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

- **🎯 AI-Ready Files:**
  All files for AI interpretation and enrichment are in
  `/Users/steveallison/Downloads/Course_content_compiler/output/summary/`

---

## **Setup**

1. **Activate your environment**

   ```sh
   conda activate course_pipeline
   ```

2. **Install all requirements** _(Enhanced with audio libraries!)_

   ```sh
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm

   # Install system dependencies
   # macOS:
   brew install ffmpeg tesseract portaudio libmagic

   # Linux:
   sudo apt-get install ffmpeg tesseract-ocr portaudio19-dev libmagic1

   # For advanced PDF table extraction:
   pip install "camelot-py[cv]" pdfplumber pillow
   ```

   _(See enhanced `requirements.txt` for complete audio library list)_

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

  **🔍 Phase 3: Comprehensive Analysis** _(ENHANCED!)_ 6. **Automated quality assessment and optimization recommendations** 7. **🎵 Advanced audio feature extraction and analysis**

---

## **Key Output Folders**

- **🎯 `/output/summary/` - AI-READY FILES**

  **Core Summary Files:**

  - **instructional_summary.jsonl / .csv:**
    - **Flattens all nested instructional content:** every slide, page, section, and component is a row
    - All extracted fields (text, tags, images, tables, charts, speaker notes, etc.)
  - **master_summary.json:** Complete aggregated content with entities and topics
  - **entities_summary.json:** Named entity extraction results
  - **topics_summary.json:** Topic modeling and classification results

  **Glossary Optimization:**

  - **glossary/summary_glossary.json / .md:** Deduplicated glossary with sample contexts
  - **glossary/term_usage.json:** Term frequency analysis
  - **glossary/unused_terms.json:** Terms to review/remove
  - **glossary/ai_prompt_templates.txt:** AI-ready enhancement prompts

  **Quality Analysis:**

  - **analysis/segment_quality_report.json:** Content quality metrics and warnings
  - **analysis/topic_coverage_report.json:** Topic distribution analysis
  - **analysis/duplicate_segments.json:** AI-detected similar content pairs
  - **analysis/topic_map.json:** Fuzzy topic mapping results

  **🎵 Advanced Audio Analysis:**

  - **analysis/audio_features/audio_analysis_summary.json:** Overall audio insights and recommendations
  - **analysis/audio_features/audio_prepped/\*.json:** Individual audio file analyses
  - **analysis/audio_features/video_prepped/\*.json:** Video audio track analyses

  **Prompt Files:**

  - **prompts/prompt_bundle.md:** Consolidated AI-ready improvement prompts
  - **prompts/topic\_\*.json:** Individual topic files for focused analysis

- **`/output/processed/` - Raw Processed Content**

  - **instructional_json/:** Hierarchical JSON for each doc, plus all slide/page images (in `/images/`)
  - **caption_prepped/:** Semantically processed VTT/SRT files
  - **audio_prepped/ & video_prepped/:** Processed media files
  - **media_metadata/:** Media file metadata (from prepped files)

- **`/output/logs/` - Pipeline Logs**

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

### **🎵 Advanced Audio Analysis**

- **Quality Assessment:** SNR, dynamic range, spectral analysis
- **Engagement Metrics:** Pitch variation, volume dynamics, speaking rate
- **Accessibility:** Speaking rate assessment, clarity, frequency analysis
- **Content Structure:** Automatic segment boundary detection
- **Speaker Analysis:** Voice consistency, characteristics analysis
- **Audio Fingerprinting:** Duplicate detection using spectral features

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
- **🎵 Comprehensive Audio Feature Analysis**
  Extracts 40+ audio features using librosa, torchaudio, Pydub, and enhanced Whisper
- **AI Prompt Templates for Glossary Review**
  Generates ready-to-use prompt files to feed AI tools for improving glossary definitions or examples.
- **Project Output Manifest**
  A top-level JSON that indexes all output artifacts for use in dashboards or LLM agents.
- **Parallel Enrichment with Thread Pooling**
  Accelerates segment processing using `--max_workers` to distribute tasks across threads.
- **Function-Level Disk Caching**
  Avoids redundant processing by caching results of enrichment functions like summarization and glossary extraction.

---

## **🎵 Audio Library Capabilities**

### **Librosa Features**

- **Spectral Analysis:** MFCCs, chroma features, spectral contrast, tonnetz
- **Tempo & Rhythm:** Beat tracking, tempo estimation, onset detection
- **Audio Quality:** Spectral centroid, rolloff, zero-crossing rate
- **Structure Analysis:** Novelty detection, segment boundaries

### **Torchaudio Features**

- **Neural Processing:** PyTorch-based audio transformations
- **Advanced Models:** Speaker embeddings, audio classification
- **High-Quality Resampling:** Professional audio processing

### **Pydub Features**

- **Format Handling:** Robust audio I/O and conversion
- **Silence Detection:** Smart speech activity detection
- **Audio Effects:** Volume normalization, filtering

### **Enhanced Whisper Features**

- **Word-Level Timing:** Precise timestamp extraction
- **Confidence Scores:** Quality assessment of transcription
- **Language Detection:** Automatic language identification
- **Multiple Model Sizes:** Tiny to Large models for different use cases

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
  → **Automated suite running all 6 analysis modules:**
  - Source distribution analysis
  - Content quality and topic analysis
  - Glossary usage optimization
  - Duplicate content detection
  - **🎵 Advanced audio feature analysis**
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
- `scripts/run_advanced_audio_analysis.py`
  → **🎵 Comprehensive audio feature extraction and quality assessment**
- `scripts/generate_output_manifest.py`
  → Indexes all pipeline outputs into a single manifest

---

## **Business Value & Use Cases**

### **Content Portfolio Optimization**

- **Quality Control:** Automated identification of content needing improvement
- **Duplicate Elimination:** Find redundant materials for consolidation
- **Glossary Cleanup:** Remove unused terms to reduce cognitive load
- **🎵 Audio Quality Assurance:** Identify recordings needing re-recording or enhancement

### **Instructional Design Insights**

- **Topic Gap Analysis:** Identify under-covered topics
- **Content Balance:** Track mix of formats (video, slides, text)
- **Learning Path Optimization:** Use duplicate detection to create learning progressions
- **🎵 Engagement Optimization:** Use audio metrics to improve delivery quality

### **AI-Powered Content Enhancement**

- **Automated Improvement:** Generated prompts ready for ChatGPT/Claude
- **Scalable Reviews:** Process hundreds of documents with consistent quality metrics
- **Smart Recommendations:** Data-driven suggestions for content creators
- **🎵 Audio Accessibility:** Automated assessment of speaking rate, clarity, and accessibility

---

## **Troubleshooting**

- **No output/logs?**
  - Double-check your input path in `.env` file and that files are present
  - Make sure you are running from the project root (`cd .../Course_content_compiler/`)
  - Check logs in `/output/logs/` for any tracebacks
- **Import/module errors?**
  - Make sure `PYTHONPATH` is set in `main.py` subprocess calls (already configured)
  - Ensure your conda environment is activated: `conda activate course_pipeline`
- **Images, tables, or charts missing?**
  - Ensure you have installed all advanced dependencies (see Setup)
  - The extractor includes speaker notes, tables, charts, and image metadata for all supported formats
- **Analysis reports empty?**
  - Run the full pipeline first to generate content, then analysis reports will be populated
  - Check that Phase 1 and 2 completed successfully before Phase 3 analysis
- **🎵 Audio analysis failing?**
  - Ensure audio libraries are installed: `pip install librosa torchaudio pydub`
  - Check system dependencies: `brew install ffmpeg portaudio` (macOS) or `sudo apt-get install ffmpeg portaudio19-dev` (Linux)
  - Verify audio files exist in `audio_prepped/` or `video_prepped/` directories

---

## **Sample requirements.txt** _(Enhanced!)_

```
# 🎵 ADVANCED AUDIO PROCESSING LIBRARIES
librosa>=0.10.0  # Advanced audio analysis, spectral features, tempo/beat tracking
torchaudio>=2.0.0  # PyTorch audio processing, models, and transforms
pydub>=0.25.0  # Audio manipulation, format conversion, effects
openai-whisper>=20230314  # Enhanced speech-to-text with word-level timing
scipy>=1.9.0  # Scientific computing for signal processing
numpy>=1.21.0  # Numerical computing foundation
soundfile>=0.12.0  # Audio I/O for librosa
ffmpeg-python>=0.2.0  # Python bindings for FFmpeg

# Core Dependencies
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

**You now have a robust, production-grade, fully modular learning content pipeline with comprehensive quality analysis, AI-powered optimization recommendations, and advanced audio feature extraction using industry-leading libraries. All AI-ready files are organized in the `/summary/` folder for easy access and upload to AI tools.
Happy compiling!**
