"""Compare the local PDF collection with the dataset metadata catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def title_key(value: str) -> str:
    """Normalize titles so punctuation and accents do not prevent a match."""
    without_copy_suffix = re.sub(r"\s+\(\d+\)$", "", value.strip())
    ascii_title = unicodedata.normalize("NFKD", without_copy_suffix)
    ascii_title = ascii_title.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", ascii_title.casefold())


def load_catalog(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Catalog not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Catalog is not valid JSON ({path}): {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("Catalog must contain a JSON list of documents.")

    invalid_rows = [
        str(index)
        for index, row in enumerate(data, start=1)
        if not isinstance(row, dict) or not isinstance(row.get("title"), str) or not row["title"].strip()
    ]
    if invalid_rows:
        raise ValueError("Missing or invalid title in catalog row(s): " + ", ".join(invalid_rows))
    return data


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_inventory(data_dir: Path, catalog_path: Path) -> tuple[list[str], bool]:
    catalog = load_catalog(catalog_path)
    pdfs = sorted(
        (path for path in data_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.casefold(),
    ) if data_dir.is_dir() else []

    catalog_counts = Counter(title_key(row["title"]) for row in catalog)
    pdf_counts = Counter(title_key(path.stem) for path in pdfs)
    lines = [f"Catálogo: {len(catalog)} registros", f"PDF locales: {len(pdfs)}"]
    issues = False

    repeated_titles = defaultdict(list)
    for row in catalog:
        repeated_titles[title_key(row["title"])].append(row["title"].strip())
    repeated_titles = {key: names for key, names in repeated_titles.items() if key and len(names) > 1}
    if repeated_titles:
        lines.append("Títulos repetidos en el catálogo:")
        for names in repeated_titles.values():
            lines.append(f"  - {names[0]} ({len(names)} registros)")

    missing = catalog_counts - pdf_counts
    if missing:
        issues = True
        lines.append("Registros del catálogo sin PDF correspondiente:")
        for row in catalog:
            key = title_key(row["title"])
            if missing[key]:
                lines.append(f"  - {row['title']}")
                missing[key] -= 1

    unlisted = pdf_counts - catalog_counts
    if unlisted:
        issues = True
        lines.append("PDF sin registro correspondiente en el catálogo:")
        for path in pdfs:
            key = title_key(path.stem)
            if unlisted[key]:
                lines.append(f"  - {path.name}")
                unlisted[key] -= 1

    hashes = defaultdict(list)
    for path in pdfs:
        hashes[sha256(path)].append(path.name)
    exact_duplicates = [names for names in hashes.values() if len(names) > 1]
    if exact_duplicates:
        issues = True
        lines.append("PDF idénticos por contenido (SHA-256):")
        for names in exact_duplicates:
            lines.append("  - " + " = ".join(names))

    if not issues:
        lines.append("Inventario consistente: cada registro corresponde a un PDF y no hay copias idénticas.")
    return lines, issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare PDF filenames and exact duplicates with the dataset catalog."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing PDFs (default: repository data/).",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset_metadata.json",
        help="JSON catalog (default: data/dataset_metadata.json).",
    )
    args = parser.parse_args()

    try:
        lines, issues = inspect_inventory(args.data_dir, args.catalog)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print("\n".join(lines))
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
