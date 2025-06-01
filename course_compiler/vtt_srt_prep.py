import webvtt
import srt
import re
import json
from pathlib import Path

from course_compiler.nlp_tagging import tag_text_components
import spacy

# For advanced named entity and keyword extraction
nlp = spacy.load("en_core_web_sm")

def parse_vtt_segments(vtt_path):
    segments = []
    try:
        for caption in webvtt.read(str(vtt_path)):
            segments.append({
                "start": caption.start,
                "end": caption.end,
                "text": caption.text.strip()
            })
    except Exception as e:
        print(f"Error parsing {vtt_path}: {e}")
    return segments

def parse_srt_segments(srt_path):
    segments = []
    try:
        with open(srt_path, "r", encoding="utf8") as f:
            srt_data = f.read()
        for caption in srt.parse(srt_data):
            segments.append({
                "start": str(caption.start),
                "end": str(caption.end),
                "text": caption.content.strip()
            })
    except Exception as e:
        print(f"Error parsing {srt_path}: {e}")
    return segments

def clean_caption_text(segments):
    cleaned = []
    for seg in segments:
        text = seg["text"]
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        cleaned.append({**seg, "cleaned_text": text})
    return cleaned

def extract_named_entities(text):
    """Extract named entities (ORG, PRODUCT, PERSON, EVENT, etc.) from text using spaCy."""
    doc = nlp(text)
    return list({ent.text for ent in doc.ents if ent.label_ in {"ORG", "PRODUCT", "PERSON", "EVENT"}})

def extract_keywords(text, top_n=5):
    """Extract keywords with a simple noun chunk approach (upgrade with KeyBERT if needed)."""
    doc = nlp(text)
    # Use noun chunks and high-frequency nouns
    keywords = {chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 1}
    keywords.update({token.lemma_ for token in doc if token.pos_ == "NOUN" and len(token) > 2})
    return list(keywords)[:top_n]

def detect_speaker(text):
    """Detect speaker label (common in meeting captions) - returns speaker or None."""
    match = re.match(r"^\s*([A-Za-z0-9_ -]+):\s*(.+)", text)
    if match:
        return match.group(1)
    return None

def detect_action_prompt(text):
    """Detect action/motivation/prompt phrases using pattern rules."""
    prompts = []
    if re.search(r"\b(try this|let's|consider|reflect|pause and|your turn|practice|discuss|please|write down|share)\b", text, re.IGNORECASE):
        prompts.append("action_prompt")
    if re.search(r"\b(great job|excellent|keep going|well done|good work)\b", text, re.IGNORECASE):
        prompts.append("motivation")
    return prompts

def semantic_process_segments(segments, use_llm=False):
    """
    For each caption segment:
    - Tags for instructional role (using rules, spaCy, and optionally LLM)
    - Extracts named entities and keywords
    - Detects speaker label if present
    - Flags action or motivational prompts
    Returns enriched segment dicts.
    """
    processed = []
    for seg in segments:
        text = seg.get("cleaned_text", seg["text"])
        tag_result = tag_text_components(text, use_llm=use_llm)
        entities = extract_named_entities(text)
        keywords = extract_keywords(text)
        speaker = detect_speaker(text)
        action_tags = detect_action_prompt(text)
        out = {**seg, **tag_result}
        if entities:
            out["entities"] = entities
        if keywords:
            out["keywords"] = keywords
        if speaker:
            out["speaker"] = speaker
        if action_tags:
            out["action_tags"] = action_tags
        processed.append(out)
    return processed

def save_caption_segments(segments, output_path):
    with open(output_path, "w") as f:
        json.dump(segments, f, indent=2)
