"""Extract selectable text from one PDF and preserve page boundaries.

Run from the repository root:
    python src/ingestion/extract_pdf.py data/example.pdf
    python src/ingestion/extract_pdf.py data/example.pdf --output data/extracted/example.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from pypdf import PdfReader
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'pypdf'. Install project dependencies with: "
        "pip install -r requirements.txt"
    ) from exc


def extract_pdf(pdf_path: Path) -> dict[str, Any]:
    """Return document metadata and extracted text, one entry per page."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    if reader.is_encrypted:
        raise ValueError(f"PDF is encrypted and cannot be read without a password: {pdf_path}")

    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append({"page": page_number, "text": text})

    metadata = reader.metadata
    return {
        "source_file": pdf_path.name,
        "page_count": len(pages),
        "metadata": {
            "title": metadata.title if metadata else None,
            "author": metadata.author if metadata else None,
        },
        "pages": pages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract text from one PDF, preserving page numbers."
    )
    parser.add_argument("pdf", type=Path, help="PDF file to process")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Write JSON to this path; by default, print it to the terminal.",
    )
    args = parser.parse_args()

    try:
        result = extract_pdf(args.pdf)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    empty_pages = [page["page"] for page in result["pages"] if not page["text"]]
    if empty_pages:
        print(
            "Warning: no selectable text found on page(s) "
            f"{', '.join(map(str, empty_pages))}. These may be scanned pages; OCR is not applied.",
            file=sys.stderr,
        )

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Extracted {result['page_count']} pages to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
