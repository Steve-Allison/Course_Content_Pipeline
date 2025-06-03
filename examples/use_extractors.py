#!/usr/bin/env python3
"""
Example script demonstrating how to use the file extractors.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from course_compiler.extractors import extract_content, register_extractor
from course_compiler.logging_utils import setup_logger

# Set up logging
logger = setup_logger(__name__)

def print_extraction_result(result: Dict[str, Any], output_format: str = 'pretty') -> None:
    """Print the extraction result in the specified format."""
    if output_format == 'json':
        print(json.dumps(result, indent=2, default=str))
    else:
        # Pretty print
        print("\n" + "=" * 80)
        print(f"Extracted content from: {result.get('metadata', {}).get('file_name', 'Unknown')}")
        print("=" * 80)

        # Print basic metadata
        meta = result.get('metadata', {})
        print(f"\nMetadata:")
        print(f"  - File: {meta.get('file_name')}")
        print(f"  - Size: {meta.get('file_size', 0) / 1024:.2f} KB")
        print(f"  - Type: {result.get('file_type', 'unknown')}")

        # Print content summary based on file type
        file_type = result.get('file_type', '').lower()

        if file_type == 'docx':
            print(f"\nDocument contains {len(result.get('paragraphs', []))} paragraphs and {len(result.get('tables', []))} tables")
            if 'paragraphs' in result and result['paragraphs']:
                print("\nFirst paragraph:")
                print("  " + "\n  ".join(result['paragraphs'][0].split('\n')[:3]) + "...")

        elif file_type == 'pptx':
            print(f"\nPresentation contains {len(result.get('slides', []))} slides")
            if 'slides' in result and result['slides']:
                slide = result['slides'][0]
                print(f"\nFirst slide: {slide.get('title', 'Untitled')}")
                if 'shapes' in slide and slide['shapes']:
                    for shape in slide['shapes'][:3]:  # Show first 3 shapes
                        if shape.get('type') == 'text':
                            print(f"  - Text: {shape.get('content', '')[:100]}...")

        elif file_type == 'pdf':
            print(f"\nPDF contains {result.get('page_count', 0)} pages")
            if 'pages' in result and result['pages']:
                page = result['pages'][0]
                print(f"\nFirst page text preview:")
                print("  " + "\n  ".join(page.get('plain_text', '').split('\n')[:5]) + "...")

        elif file_type == 'xlsx':
            print(f"\nExcel workbook contains {len(result.get('worksheets', []))} worksheets")
            if 'worksheets' in result and result['worksheets']:
                for i, ws in enumerate(result['worksheets'][:3]):  # Show first 3 worksheets
                    print(f"\nWorksheet {i+1}: {ws.get('name')}")
                    print(f"  - Dimensions: {ws.get('dimensions', 'N/A')}")
                    print(f"  - Rows: {ws.get('row_count', 0)}, Columns: {ws.get('column_count', 0)}")
                    print(f"  - Tables: {len(ws.get('tables', []))}")

                    # Show a small preview of the data
                    if 'rows' in ws and ws['rows']:
                        print("\n  First few rows:")
                        for row in ws['rows'][:5]:  # Show first 5 rows
                            # Get non-None cell values
                            cell_values = [str(cell.get('value', '')) 
                                         for cell in row.get('cells', []) 
                                         if cell and cell.get('value') is not None]
                            if cell_values:
                                print(f"    {' | '.join(cell_values[:8])}" + 
                                      ("..." if len(cell_values) > 8 else ""))

        elif file_type == 'text':
            print(f"\nText content ({len(result.get('content', ''))} characters):")
            print("  " + "\n  ".join(result.get('content', '')[:200].split('\n')) + "...")

        print("\n" + "=" * 80 + "\n")

def process_file(file_path: str, output_format: str = 'pretty') -> bool:
    """Process a single file and print the results."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {file_path}")
        return False

    try:
        logger.info(f"Extracting content from: {path}")
        result = extract_content(path)
        print_extraction_result(result, output_format)
        return True
    except Exception as e:
        logger.error(f"Error processing {path}: {e}", exc_info=True)
        return False

def process_directory(directory: str, output_format: str = 'pretty') -> None:
    """Process all supported files in a directory."""
    path = Path(directory)
    if not path.exists() or not path.is_dir():
        logger.error(f"Directory not found: {directory}")
        return

    # Get all files in the directory
    files = [f for f in path.glob('*') if f.is_file()]
    if not files:
        logger.warning(f"No files found in directory: {directory}")
        return

    # Process each file
    success_count = 0
    for file_path in files:
        if process_file(str(file_path), output_format):
            success_count += 1

    print(f"\nProcessed {success_count} of {len(files)} files successfully.")

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Extract content from files')
    parser.add_argument('path', help='File or directory path to process')
    parser.add_argument('--format', choices=['pretty', 'json'], default='pretty',
                       help='Output format (default: pretty)')

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path not found: {args.path}")
        return 1

    if path.is_file():
        success = process_file(args.path, args.format)
        return 0 if success else 1
    else:
        process_directory(args.path, args.format)
        return 0

if __name__ == '__main__':
    sys.exit(main())
