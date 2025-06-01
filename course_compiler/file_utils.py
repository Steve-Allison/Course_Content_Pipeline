from pathlib import Path
import re

def ensure_dirs(base, subfolders=("instructional_json",)):
    for folder in subfolders:
        (Path(base) / folder).mkdir(parents=True, exist_ok=True)

def sanitize_filename(filename):
    """
    Sanitizes a filename by replacing spaces with underscores and
    removing problematic characters except for alphanumerics, underscores,
    hyphens, and dots. Safe for Mac and cross-platform usage.
    """
    # Replace all spaces with underscores
    filename = filename.replace(" ", "_")
    # Remove any character that is not alphanumeric, underscore, dot, or hyphen
    filename = re.sub(r'[^\w\.-]', '', filename)
    return filename
