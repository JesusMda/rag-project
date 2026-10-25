"""Build and query a small local BM25 index over chunk JSON files.

Examples from the repository root:
    python src/retrieval/bm25.py build data/chunks/*.json
    python src/retrieval/bm25.py search "MRtrix tractography" --index data/index/bm25.json

The index contains extracted text, so keep it in an ignored local data path.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = PROJECT_ROOT / "data" / "index" / "bm25.json"
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Return case and accent normalized word tokens; retain technical acronyms."""
    normalized = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return TOKEN_PATTERN.findall(normalized)


def load_chunks(paths: list[Path]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid chunk JSON in {path}: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array of chunks in {path}.")
        for row_number, row in enumerate(payload, start=1):
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"Chunk {row_number} in {path} must have a text string.")
            chunk = dict(row)
            chunk.setdefault("chunk_id", f"{path.name}:{row_number}")
            if not isinstance(chunk["chunk_id"], str) or not chunk["chunk_id"]:
                raise ValueError(f"Chunk {row_number} in {path} has an invalid chunk_id.")
            if chunk["chunk_id"] in seen_ids:
                raise ValueError(f"Duplicate chunk_id: {chunk['chunk_id']}")
            seen_ids.add(chunk["chunk_id"])
            chunks.append(chunk)
    if not chunks:
        raise ValueError("No chunks found in the supplied files.")
    return chunks


def build_index(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    documents = []
    document_frequency: Counter[str] = Counter()
    total_length = 0
    for chunk in chunks:
        terms = tokenize(chunk["text"])
        frequencies = Counter(terms)
        document_frequency.update(frequencies.keys())
        total_length += len(terms)
        documents.append({"chunk": chunk, "terms": dict(frequencies), "length": len(terms)})
    return {
        "format_version": 1,
        "k1": K1,
        "b": B,
        "document_count": len(documents),
        "average_length": total_length / len(documents),
        "document_frequency": dict(document_frequency),
        "documents": documents,
    }


def save_index(index: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def query_index(index: dict[str, Any], query: str, limit: int = 5) -> list[dict[str, Any]]:
    query_terms = tokenize(query)
    if not query_terms:
        raise ValueError("The query has no searchable words.")
    count = index["document_count"]
    average_length = index["average_length"] or 1.0
    frequencies = index["document_frequency"]
    scores = []
    for document in index["documents"]:
        terms = document["terms"]
        length = document["length"]
        score = 0.0
        for term in query_terms:
            tf = terms.get(term, 0)
            if not tf:
                continue
            df = frequencies.get(term, 0)
            inverse_frequency = math.log(1 + (count - df + 0.5) / (df + 0.5))
            denominator = tf + index["k1"] * (1 - index["b"] + index["b"] * length / average_length)
            score += inverse_frequency * tf * (index["k1"] + 1) / denominator
        if score > 0:
            chunk = document["chunk"]
            scores.append({
                "score": score,
                "chunk_id": chunk.get("chunk_id"),
                "title": chunk.get("title") or chunk.get("source_file") or "Título desconocido",
                "pages": chunk.get("pages") or list(range(
                    chunk.get("start_page", 1), chunk.get("end_page", chunk.get("start_page", 1)) + 1
                )),
                "text": chunk["text"],
            })
    scores.sort(key=lambda result: (-result["score"], str(result["chunk_id"])))
    return scores[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description="Local BM25 search over page-aware text chunks.")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Build a BM25 index from chunk JSON files.")
    build_parser.add_argument("chunks", nargs="+", type=Path, help="Chunk JSON files.")
    build_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Output index path.")
    search_parser = commands.add_parser("search", help="Search a previously built BM25 index.")
    search_parser.add_argument("query", help="Words or acronyms to search for.")
    search_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Index JSON path.")
    search_parser.add_argument("--limit", type=int, default=5, help="Maximum results (default: 5).")
    args = parser.parse_args()

    try:
        if args.command == "build":
            index = build_index(load_chunks(args.chunks))
            save_index(index, args.index)
            print(f"Indexado {index['document_count']} fragmentos en {args.index}")
            return 0
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        index = json.loads(args.index.read_text(encoding="utf-8"))
        if index.get("format_version") != 1:
            raise ValueError("Unsupported or invalid BM25 index format.")
        results = query_index(index, args.query, args.limit)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No se encontraron fragmentos para esa consulta.")
        return 0
    for rank, result in enumerate(results, start=1):
        pages = ", ".join(map(str, result["pages"])) or "desconocida"
        print(f"{rank}. {result['title']} — p. {pages} (BM25 {result['score']:.3f})")
        print(result["text"])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
