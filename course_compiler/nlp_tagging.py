import re
import spacy

try:
    from transformers import pipeline
    # Only load this ONCE! Slow to start.
    zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
except ImportError:
    zero_shot = None

INSTRUCTIONAL_LABELS = [
    "learning objective",
    "heading",
    "question",
    "example",
    "summary",
    "assessment",
    "instruction",
    "concept",
    "feedback",
    "introduction",
    "conclusion",
    # Expanded types
    "definition",
    "reference",
    "warning",
    "tip",
    "note",
    "activity",
    "exercise",
    "case study",
    "video",
    "image",
    "table",
    "chart",
    "slide",
    "page",
    "section",
]

nlp = spacy.load("en_core_web_sm")

def rule_based_tags(text):
    tags = []
    txt_lower = text.lower()
    # Instructional roles
    if re.search(r"\b(objective|goal|outcome)\b", txt_lower):
        tags.append("learning_objective")
    if txt_lower.endswith("?") or txt_lower.startswith("why") or txt_lower.startswith("how"):
        tags.append("question")
    if re.search(r"\bexample\b|\be\.g\.\b", txt_lower):
        tags.append("example")
    if "summary" in txt_lower or "recap" in txt_lower:
        tags.append("summary")
    if re.search(r"\b(quiz|assessment|test)\b", txt_lower):
        tags.append("assessment")
    if len(text.split()) < 8 and (text.isupper() or text.istitle()):
        tags.append("heading")
    if "introduction" in txt_lower or txt_lower.startswith("welcome"):
        tags.append("introduction")
    if "conclusion" in txt_lower or "thank you" in txt_lower or "summary" in txt_lower:
        tags.append("conclusion")
    # Expanded instructional types
    if re.search(r"\bfeedback\b|well done|good job|review", txt_lower):
        tags.append("feedback")
    if re.search(r"\bdefinition\b|is defined as|means", txt_lower):
        tags.append("definition")
    if re.search(r"\btip:|pro tip|hint", txt_lower):
        tags.append("tip")
    if re.search(r"\bnote:|important:|remember", txt_lower):
        tags.append("note")
    if re.search(r"\bactivity\b|try this|exercise|practice", txt_lower):
        tags.append("activity")
    if re.search(r"\bexercise\b|practice question|drill", txt_lower):
        tags.append("exercise")
    if re.search(r"case study|real-world example", txt_lower):
        tags.append("case_study")
    if re.search(r"\bvideo\b|watch|play video|see video", txt_lower):
        tags.append("video")
    if re.search(r"\bimage\b|see image|figure|diagram|illustration", txt_lower):
        tags.append("image")
    if re.search(r"\btable\b|see table|tabular", txt_lower):
        tags.append("table")
    if re.search(r"\bchart\b|graph|plot|visualization", txt_lower):
        tags.append("chart")
    if re.search(r"\bwarning\b|caution|be careful", txt_lower):
        tags.append("warning")
    if re.search(r"\breference\b|see also|related", txt_lower):
        tags.append("reference")
    # Structural tags (for context or explicit mention)
    if re.search(r"\bslide\b", txt_lower):
        tags.append("slide")
    if re.search(r"\bpage\b", txt_lower):
        tags.append("page")
    if re.search(r"\bsection\b|part|chapter", txt_lower):
        tags.append("section")
    return list(set(tags))

def spacy_based_tags(text):
    doc = nlp(text)
    tags = []
    # Imperative/instructional language
    if doc and doc[0].pos_ == "VERB":
        tags.append("instruction")
    # Named entities (could be a concept)
    for ent in doc.ents:
        if ent.label_ in {"ORG", "PRODUCT", "PERSON", "EVENT"}:
            tags.append("concept")
    return list(set(tags))

def llm_based_tags(text):
    if zero_shot is None:
        return []
    result = zero_shot(
        sequences=text,
        candidate_labels=INSTRUCTIONAL_LABELS,
        multi_label=True
    )
    tags = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.35]
    return tags

def tag_text_components(text, use_llm=True):
    """
    Returns dict: {"type": "text", "text": <original>, "tags": [roles]}
    Uses rules, spaCy, and zero-shot LLM (optional).
    """
    tags = set()
    tags.update(rule_based_tags(text))
    tags.update(spacy_based_tags(text))
    if use_llm:
        tags.update(llm_based_tags(text))
    return {"type": "text", "text": text, "tags": sorted(tags)}
