"""Build a local multilingual embedding index and compare semantic/hybrid search.

Examples from the repository root:
    python src/retrieval/semantic.py build data/chunks/*.json
    python src/retrieval/semantic.py search "¿Qué mide NODDI?" --limit 5
    python src/retrieval/semantic.py hybrid "¿Qué mide NODDI?" --limit 5

The embedding model and all indexes are local after the first model download.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise SystemExit(
        "Semantic search dependencies are missing. Install them with: "
        "pip install -r requirements.txt"
    ) from exc

try:
    from .bm25 import DEFAULT_INDEX as DEFAULT_BM25_INDEX
    from .bm25 import load_chunks, query_index
except ImportError:  # Support running this file directly from the repository root.
    from bm25 import DEFAULT_INDEX as DEFAULT_BM25_INDEX
    from bm25 import load_chunks, query_index


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_VECTOR_INDEX = PROJECT_ROOT / "data" / "index" / "semantic.npz"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "index" / "semantic.json"
RRF_K = 60
DEFAULT_SEMANTIC_WEIGHT = 3.0
DEFAULT_BM25_WEIGHT = 1.0


@lru_cache(maxsize=2)
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")


def build_vector_index(
    chunks: list[dict[str, Any]], model_name: str = MODEL_NAME, batch_size: int = 32
) -> tuple[np.ndarray, dict[str, Any]]:
    model = load_model(model_name)
    texts = [f"{chunk.get('title', '')}\n{chunk['text']}" for chunk in chunks]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    metadata = {
        "format_version": 1,
        "model": model_name,
        "document_count": len(chunks),
        "embedding_dimensions": int(vectors.shape[1]),
        "documents": chunks,
    }
    return np.asarray(vectors, dtype=np.float32), metadata


def save_vector_index(
    vectors: np.ndarray,
    metadata: dict[str, Any],
    vector_path: Path = DEFAULT_VECTOR_INDEX,
    metadata_path: Path = DEFAULT_METADATA,
) -> None:
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(vector_path, embeddings=vectors)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_vector_index(
    vector_path: Path = DEFAULT_VECTOR_INDEX, metadata_path: Path = DEFAULT_METADATA
) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(vector_path) as payload:
        vectors = np.asarray(payload["embeddings"], dtype=np.float32)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported or invalid semantic index format.")
    if len(metadata["documents"]) != len(vectors):
        raise ValueError("Semantic index metadata and vectors have different lengths.")
    return vectors, metadata


def semantic_search(
    query: str,
    vectors: np.ndarray,
    metadata: dict[str, Any],
    limit: int = 5,
) -> list[dict[str, Any]]:
    model = load_model(metadata["model"])
    query_vector = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
    )[0]
    similarities = vectors @ query_vector
    limit = min(limit, len(similarities))
    candidate_ids = np.argpartition(-similarities, limit - 1)[:limit]
    sorted_ids = sorted(candidate_ids.tolist(), key=lambda idx: (-float(similarities[idx]), idx))
    return [
        _render_result(metadata["documents"][idx], float(similarities[idx]))
        for idx in sorted_ids
    ]


def _render_result(chunk: dict[str, Any], score: float) -> dict[str, Any]:
    start = chunk.get("start_page", 1)
    end = chunk.get("end_page", start)
    pages = chunk.get("pages") or list(range(start, end + 1))
    return {
        "score": score,
        "chunk_id": chunk.get("chunk_id"),
        "title": chunk.get("title") or chunk.get("source_file") or "Título desconocido",
        "pages": pages,
        "text": chunk["text"],
    }


def hybrid_search(
    query: str,
    vectors: np.ndarray,
    metadata: dict[str, Any],
    bm25_index: dict[str, Any],
    limit: int = 5,
    candidate_limit: int = 50,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
) -> list[dict[str, Any]]:
    if semantic_weight <= 0 or bm25_weight < 0:
        raise ValueError("semantic weight must be positive and BM25 weight cannot be negative.")
    semantic_results = semantic_search(query, vectors, metadata, candidate_limit)
    lexical_results = query_index(bm25_index, query, candidate_limit)
    fused: dict[str, dict[str, Any]] = {}
    for ranking, weight in ((semantic_results, semantic_weight), (lexical_results, bm25_weight)):
        for rank, result in enumerate(ranking, start=1):
            key = str(result["chunk_id"])
            entry = fused.setdefault(key, {**result, "rrf_score": 0.0, "semantic_rank": None, "bm25_rank": None})
            entry["rrf_score"] += weight / (RRF_K + rank)
            if ranking is semantic_results:
                entry["semantic_rank"] = rank
            else:
                entry["bm25_rank"] = rank
    results = sorted(fused.values(), key=lambda item: (-item["rrf_score"], str(item["chunk_id"])))
    return results[:limit]


def print_results(results: list[dict[str, Any]], score_name: str) -> None:
    if not results:
        print("No se encontraron fragmentos para esa consulta.")
        return
    for rank, result in enumerate(results, start=1):
        pages = ", ".join(map(str, result["pages"])) or "desconocida"
        line = f"{rank}. {result['title']} — p. {pages} ({score_name} {result['score']:.3f})"
        if "rrf_score" in result:
            line = (
                f"{rank}. {result['title']} — p. {pages} (RRF {result['rrf_score']:.4f}; "
                f"semántica #{result['semantic_rank'] or '—'}, BM25 #{result['bm25_rank'] or '—'})"
            )
        print(line)
        print(result["text"])
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Local multilingual semantic and hybrid retrieval.")
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="Build a local embedding index from chunk JSON files.")
    build_parser.add_argument("chunks", nargs="+", type=Path, help="Chunk JSON files.")
    build_parser.add_argument("--model", default=MODEL_NAME, help=f"Sentence Transformers model (default: {MODEL_NAME}).")
    build_parser.add_argument("--batch-size", type=int, default=32)
    build_parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTOR_INDEX)
    build_parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)

    for command in ("search", "hybrid"):
        subparser = commands.add_parser(command, help=f"Run {command} search over the local indexes.")
        subparser.add_argument("query", help="Question or search text.")
        subparser.add_argument("--limit", type=int, default=5)
        subparser.add_argument("--vectors", type=Path, default=DEFAULT_VECTOR_INDEX)
        subparser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
        if command == "hybrid":
            subparser.add_argument("--bm25-index", type=Path, default=DEFAULT_BM25_INDEX)
            subparser.add_argument("--candidates", type=int, default=50)
            subparser.add_argument("--semantic-weight", type=float, default=DEFAULT_SEMANTIC_WEIGHT)
            subparser.add_argument("--bm25-weight", type=float, default=DEFAULT_BM25_WEIGHT)
    args = parser.parse_args()

    try:
        if args.command == "build":
            if args.batch_size < 1:
                raise ValueError("--batch-size must be at least 1.")
            chunks = load_chunks(args.chunks)
            vectors, metadata = build_vector_index(chunks, args.model, args.batch_size)
            save_vector_index(vectors, metadata, args.vectors, args.metadata)
            print(
                f"Indexados {len(chunks)} fragmentos; dimensiones: {metadata['embedding_dimensions']}; "
                f"modelo: {metadata['model']}"
            )
            print(f"Vectores: {args.vectors}\nMetadatos: {args.metadata}")
            return 0

        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        vectors, metadata = load_vector_index(args.vectors, args.metadata)
        if args.command == "search":
            print_results(semantic_search(args.query, vectors, metadata, args.limit), "coseno")
        else:
            if args.candidates < args.limit:
                raise ValueError("--candidates must be greater than or equal to --limit.")
            bm25_index = json.loads(args.bm25_index.read_text(encoding="utf-8"))
            print_results(
                hybrid_search(
                    args.query,
                    vectors,
                    metadata,
                    bm25_index,
                    args.limit,
                    args.candidates,
                    args.semantic_weight,
                    args.bm25_weight,
                ),
                "coseno",
            )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
