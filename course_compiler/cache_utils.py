from concurrent.futures import ThreadPoolExecutor

def enrich_single_segment(seg_id, text):
    summary = summarise_text(text)
    glossary_terms = extract_glossary_terms(text)
    tags = tag_text(text)
    metadata = generate_metadata(text)

    return Segment(
        id=seg_id,
        text=text,
        summary=summary,
        topics=tags.get("topics", []),
        tags=tags.get("tags", []),
        glossary_terms=glossary_terms,
        metadata=metadata
    )

def enrich_segments(segments_raw, max_workers=None):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(enrich_single_segment, seg_id, text) for seg_id, text in segments_raw]
        return [future.result() for future in futures]