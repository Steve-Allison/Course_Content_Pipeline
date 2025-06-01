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
    "conclusion"
]

nlp = spacy.load("en_core_web_sm")

def rule_based_tags(text):
    tags = []
    txt_lower = text.lower()
    if re.search(r"\b(objective|goal|outcome)\b", txt_lower):
        tags.append("learning_objective")
    if txt_lower.endswith("?") or txt_lower.startswith("why") or txt_lower.startswith("how"):
        tags.append("question")
    if re.search(r"\bexample\b|\be\.g\.", txt_lower):
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
