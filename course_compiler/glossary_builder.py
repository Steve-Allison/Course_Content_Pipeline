"""
build_glossary.py

Extracts glossary terms from instructional content using NLP and keyword models.
Patched version adds fallback logic, logging, and example usage mapping.
"""

import os
import json
import re
import logging
import argparse
from pathlib import Path
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from keybert import KeyBERT

# Setup
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
kw_model = KeyBERT()

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def extract_keywords_fallback(text, top_n=10):
    """Fallback using KeyBERT if term extraction fails or is too sparse."""
    try:
        keywords = [kw[0] for kw in kw_model.extract_keywords(text, top_n=top_n)]
        return list(set(keywords))
    except Exception as e:
        logging.warning(f"KeyBERT fallback failed: {e}")
        return []


def find_example_sentences(text, terms):
    """Map each term to the first matching sentence found in the text."""
    examples = {}
    sentences = sent_tokenize(text)
    for term in terms:
        for sentence in sentences:
            if term.lower() in sentence.lower():
                examples[term] = sentence
                break
        if term not in examples:
            examples[term] = ""
    return examples


def clean_term(term):
    """Normalize and clean extracted term."""
    return term.strip().lower()


def extract_terms_from_text(text, fallback=True):
    """Extract high-frequency tokens or fallback to KeyBERT."""
    tokens = word_tokenize(text)
    candidates = [w for w in tokens if w.lower() not in stop_words and re.search(r'\w', w)]
    freq = Counter(candidates)
    terms = [t for t, _ in freq.most_common(30) if len(t) > 2]

    if not terms and fallback:
        terms = extract_keywords_fallback(text)

    return sorted(set(clean_term(t) for t in terms if t))


def build_glossary(input_file, output_file):
    """Main routine: read input, extract terms, write glossary JSON."""
    text = Path(input_file).read_text()
    terms = extract_terms_from_text(text)

    if not terms:
        logging.warning("No glossary terms found. Using fallback seed terms.")
        terms = ["engagement", "automation", "workflow", "analytics"]

    examples = find_example_sentences(text, terms)

    glossary = []
    for term in sorted(terms):
        glossary.append({
            "term": term,
            "definition": "",  # To be filled by SME or enrichment step
            "example": examples.get(term, ""),
            "difficulty": "intermediate"
        })

    with open(output_file, "w") as f:
        json.dump(glossary, f, indent=2)

    logging.info(f"Glossary written to {output_file} with {len(glossary)} terms.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to extracted text input")
    parser.add_argument("output_file", help="Path to save glossary JSON")
    args = parser.parse_args()

    build_glossary(args.input_file, args.output_file)
