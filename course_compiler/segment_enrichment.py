from concurrent.futures import ThreadPoolExecutor
from course_compiler.nlp_tagging import tag_text_components
import uuid

class Segment:
    def __init__(self, id, text, summary=None, topics=None, tags=None, glossary_terms=None, metadata=None):
        self.id = id
        self.text = text
        self.summary = summary or ""
        self.topics = topics or []
        self.tags = tags or []
        self.glossary_terms = glossary_terms or []
        self.metadata = metadata or {}

def summarise_text(text):
    """Generate a basic summary of the text."""
    sentences = text.split('.')
    if len(sentences) <= 2:
        return text.strip()
    return sentences[0].strip() + "."

def extract_glossary_terms(text):
    """Extract potential glossary terms from text."""
    # Simple implementation - extract capitalized words and common terms
    import re
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    # Filter out common words and keep unique terms
    common_words = {'The', 'This', 'That', 'These', 'Those', 'And', 'But', 'Or', 'For', 'With'}
    terms = [word for word in set(words) if word not in common_words and len(word) > 2]
    return terms[:5]  # Return top 5 terms

def tag_text(text):
    """Tag text with topics and instructional tags."""
    result = tag_text_components(text, use_llm=False)  # Use basic tagging to avoid errors
    return {
        "topics": result.get("tags", [])[:3],  # Use tags as topics for simplicity
        "tags": result.get("tags", [])
    }

def generate_metadata(text):
    """Generate metadata for the text segment."""
    return {
        "word_count": len(text.split()),
        "character_count": len(text),
        "sentence_count": len(text.split('.')),
        "processed": True
    }

def enrich_single_segment(segment_data):
    """Enrich a single segment with summary, tags, and metadata."""
    if isinstance(segment_data, dict):
        seg_id = segment_data.get("id", str(uuid.uuid4()))
        text = segment_data.get("text", "")
    else:
        seg_id = str(uuid.uuid4())
        text = str(segment_data)

    summary = summarise_text(text)
    glossary_terms = extract_glossary_terms(text)
    tags_data = tag_text(text)
    metadata = generate_metadata(text)

    return {
        "id": seg_id,
        "text": text,
        "summary": summary,
        "topics": tags_data.get("topics", []),
        "tags": tags_data.get("tags", []),
        "glossary_terms": glossary_terms,
        "metadata": metadata
    }

def enrich_segments(segments_raw, max_workers=None):
    """Enrich multiple segments in parallel."""
    if not segments_raw:
        return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(enrich_single_segment, segment) for segment in segments_raw]
        return [future.result() for future in futures]
