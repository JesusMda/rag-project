"""Clean extracted PDF text and split it into page-aware chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def title_key(value: str) -> str:
    """Normalize titles for matching PDF filenames to catalog records."""
    value = re.sub(r"\s+\(\d+\)$", "", value.strip())
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def catalog_metadata(source_file: str, catalog_path: Path) -> dict[str, Any]:
    rows = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("The metadata catalog must be a JSON list.")

    source_title = Path(source_file).stem
    key = title_key(source_title)
    matches = [row for row in rows if isinstance(row, dict) and title_key(row.get("title", "")) == key]
    if not matches:
        raise ValueError(f"No catalog entry matches the PDF title: {source_title}")

    fields = ("title", "authors", "year", "doi")
    first = {field: matches[0].get(field) for field in fields}
    if any({field: row.get(field) for field in fields} != first for row in matches[1:]):
        raise ValueError(f"Several conflicting catalog entries match: {source_title}")
    return first


def load_pages(extraction_path: Path) -> tuple[str, list[dict[str, Any]]]:
    document = json.loads(extraction_path.read_text(encoding="utf-8"))
    source_file = document.get("source_file")
    pages = document.get("pages")
    if not isinstance(source_file, str) or not isinstance(pages, list):
        raise ValueError("Extraction JSON must contain 'source_file' and a 'pages' list.")

    normalized_pages = []
    for item in pages:
        if not isinstance(item, dict) or not isinstance(item.get("page"), int):
            raise ValueError("Each page must contain an integer 'page' number and 'text'.")
        if not isinstance(item.get("text"), str):
            raise ValueError(f"Page {item['page']} has no text string.")
        normalized_pages.append({"page": item["page"], "text": item["text"]})
    return source_file, normalized_pages


def repeated_edge_lines(pages: list[dict[str, Any]], title: str) -> set[str]:
    """Find likely running headers/footers repeated on many pages."""
    if len(pages) < 4:
        return set()

    occurrences: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in page["text"].splitlines() if line.strip()]
        edge_lines = set(lines[:2] + lines[-2:])
        for line in edge_lines:
            key = re.sub(r"\s+", " ", line).casefold()
            if len(key) < 4 or len(key) > 180 or title_key(key) == title_key(title):
                continue
            if re.fullmatch(r"[\d\s().ivxlcdmIVXLCDM-]+", line):
                continue
            occurrences[key] += 1

    threshold = max(3, math.ceil(len(pages) * 0.5))
    return {line for line, count in occurrences.items() if count >= threshold}


def clean_page(text: str, repeated_lines: set[str]) -> str:
    text = text.replace("\x00", "").replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Join words split by a line-ending hyphen, while preserving normal hyphens.
    text = re.sub(r"(?<=[A-Za-zÀ-ÖØ-öø-ÿ])-\s*\n\s*(?=[a-zà-öø-ÿ])", "", text)

    kept_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        key = re.sub(r"\s+", " ", stripped).casefold()
        if key in repeated_lines:
            continue
        if re.fullmatch(r"\s*(?:\d+|[ivxlcdm]+)\s*", stripped, flags=re.IGNORECASE):
            continue
        kept_lines.append(stripped)

    paragraphs = []
    current = []
    for line in kept_lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(re.sub(r"\s+", " ", line))
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs).strip()


def split_long_text(text: str, max_chars: int) -> list[str]:
    """Split oversized paragraphs near sentence or word boundaries."""
    parts = []
    remainder = text.strip()
    while len(remainder) > max_chars:
        limit = max_chars
        floor = max(1, int(max_chars * 0.6))
        candidates = [
            remainder.rfind(mark, floor, limit)
            for mark in (". ", "? ", "! ", "; ", ": ", " ")
        ]
        boundary = max(candidates)
        if boundary < floor:
            boundary = limit
        elif remainder[boundary : boundary + 2] in {". ", "? ", "! ", "; ", ": "}:
            boundary += 1
        part = remainder[:boundary].strip()
        if not part:
            part = remainder[:limit]
            boundary = limit
        parts.append(part)
        remainder = remainder[boundary:].strip()
    if remainder:
        parts.append(remainder)
    return parts


def part_length(parts: list[dict[str, Any]]) -> int:
    return sum(len(part["text"]) for part in parts) + max(0, len(parts) - 1) * 2


def overlap_tail(parts: list[dict[str, Any]], overlap_chars: int) -> list[dict[str, Any]]:
    if overlap_chars <= 0:
        return []
    tail = []
    remaining = overlap_chars
    for part in reversed(parts):
        if remaining <= 0:
            break
        text = part["text"]
        selected = text[-remaining:]
        tail.insert(0, {"text": selected, "pages": set(part["pages"])})
        remaining -= len(selected) + 2
    return tail


def render_parts(parts: list[dict[str, Any]]) -> str:
    return "\n\n".join(part["text"] for part in parts).strip()


def make_chunks(
    pages: list[dict[str, Any]],
    title: str,
    source_file: str,
    metadata: dict[str, Any],
    chunk_size: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    repeated = repeated_edge_lines(pages, title)
    segments = []
    for page in pages:
        cleaned = clean_page(page["text"], repeated)
        for paragraph in re.split(r"\n\s*\n", cleaned):
            for part in split_long_text(paragraph.strip(), chunk_size):
                segments.append({"text": part, "pages": {page["page"]}})

    document_id = hashlib.sha256(source_file.encode("utf-8")).hexdigest()[:12]
    chunks = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = render_parts(current)
        source_pages = sorted(set().union(*(part["pages"] for part in current)))
        if text:
            index = len(chunks) + 1
            chunks.append(
                {
                    "chunk_id": f"{document_id}:chunk-{index:04d}",
                    "document_id": document_id,
                    "source_file": source_file,
                    **metadata,
                    "start_page": source_pages[0],
                    "end_page": source_pages[-1],
                    "pages": source_pages,
                    "text": text,
                }
            )
        current = overlap_tail(current, overlap_chars)

    for segment in segments:
        candidate = current + [segment]
        if current and part_length(candidate) > chunk_size:
            flush()
            candidate = current + [segment]
            # If a long next segment cannot fit beside the overlap, drop the
            # overlap rather than emitting an overlap-only chunk.
            if part_length(candidate) > chunk_size:
                candidate = [segment]
        current = candidate
    if current:
        text = render_parts(current)
        source_pages = sorted(set().union(*(part["pages"] for part in current)))
        if text:
            index = len(chunks) + 1
            chunks.append(
                {
                    "chunk_id": f"{document_id}:chunk-{index:04d}",
                    "document_id": document_id,
                    "source_file": source_file,
                    **metadata,
                    "start_page": source_pages[0],
                    "end_page": source_pages[-1],
                    "pages": source_pages,
                    "text": text,
                }
            )
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Clean extracted PDF text and split it into page-aware chunks."
    )
    parser.add_argument("extraction_json", type=Path, help="JSON produced by extract_pdf.py")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset_metadata.json",
        help="Dataset metadata catalog (default: data/dataset_metadata.json).",
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Write the chunk array here; default is stdout."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1200, help="Maximum chunk length in characters."
    )
    parser.add_argument(
        "--overlap", type=int, default=150, help="Approximate overlap between chunks in characters."
    )
    args = parser.parse_args()
    if args.chunk_size < 1 or args.overlap < 0 or args.overlap >= args.chunk_size:
        parser.error("chunk-size must be positive and overlap must be >= 0 and < chunk-size")

    try:
        source_file, pages = load_pages(args.extraction_json)
        metadata = catalog_metadata(source_file, args.catalog)
        chunks = make_chunks(
            pages,
            metadata["title"],
            source_file,
            metadata,
            args.chunk_size,
            args.overlap,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rendered = json.dumps(chunks, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Created {len(chunks)} chunks at {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
