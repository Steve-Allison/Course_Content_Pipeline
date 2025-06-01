try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None

def extract_ocr_from_image(img_path):
    """
    Extract text from an image using pytesseract.
    Returns OCR'd text, or empty string if not available.
    """
    if pytesseract is None:
        return ""
    try:
        text = pytesseract.image_to_string(Image.open(img_path))
        return text.strip()
    except Exception:
        return ""
