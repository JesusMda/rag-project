"""Answer questions from retrieved corpus chunks using a local Ollama model.

Examples from the repository root:
    python src/retrieval/answering.py "¿Qué mide NODDI?"
    python src/retrieval/answering.py "¿Qué mide NODDI?" --limit 8 --model llama3.1

The question and retrieved text are sent only to the configured Ollama server.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from .semantic import (
        DEFAULT_BM25_INDEX,
        DEFAULT_METADATA,
        DEFAULT_VECTOR_INDEX,
        hybrid_search,
        load_vector_index,
    )
except ImportError:  # Support running this file directly from the repository root.
    from semantic import (
        DEFAULT_BM25_INDEX,
        DEFAULT_METADATA,
        DEFAULT_VECTOR_INDEX,
        hybrid_search,
        load_vector_index,
    )


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"
CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


def build_context(results: list[dict[str, Any]], max_chars: int) -> tuple[str, list[dict[str, Any]]]:
    """Render bounded evidence and retain only sources actually passed to the model."""
    blocks = []
    sources = []
    used_chars = 0
    for result in results:
        text = result.get("text", "").strip()
        if not text:
            continue
        remaining = max_chars - used_chars
        if remaining <= 0:
            break
        text = text[:remaining]
        source_id = f"S{len(sources) + 1}"
        pages = result.get("pages") or [result.get("start_page", 1)]
        title = result.get("title") or "Título desconocido"
        authors = result.get("authors")
        year = result.get("year")
        doi = result.get("doi")
        source = {
            "id": source_id,
            "title": title,
            "authors": authors,
            "year": year,
            "doi": doi,
            "pages": pages,
            "chunk_id": result.get("chunk_id"),
        }
        sources.append(source)
        citation = f"[{source_id}] {title}, páginas {', '.join(map(str, pages))}"
        if authors:
            citation += f"; autores: {authors}"
        if year:
            citation += f"; año: {year}"
        if doi:
            citation += f"; DOI: {doi}"
        blocks.append(f"{citation}\nEvidencia:\n{text}")
        used_chars += len(text)
    return "\n\n---\n\n".join(blocks), sources


def ask_ollama(question: str, context: str, base_url: str, model: str) -> str:
    """Generate an answer through Ollama's local chat API."""
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Responde en español usando exclusivamente la evidencia proporcionada. "
                    "Cada afirmación factual debe llevar una o más citas con los identificadores "
                    "exactos [S1], [S2], etc. No inventes datos ni citas. Distingue claramente "
                    "una inferencia de lo que el texto afirma. Si la evidencia es insuficiente, "
                    "contradictoria o no responde la pregunta, dilo explícitamente y no completes "
                    "la respuesta con conocimiento externo."
                ),
            },
            {
                "role": "user",
                "content": f"Pregunta:\n{question}\n\nFragmentos recuperados:\n{context}",
            },
        ],
        "options": {"temperature": 0.1},
    }
    request = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama devolvió HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            f"No se pudo conectar con Ollama en {base_url}. Inicia Ollama y confirma que el modelo '{model}' está disponible."
        ) from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama no devolvió una respuesta válida: {exc}") from exc
    try:
        return body["message"]["content"].strip()
    except (KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError("La respuesta de Ollama no contiene message.content.") from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Answer with a local Ollama model using only retrieved corpus evidence."
    )
    parser.add_argument("question", help="Question to answer from the local document corpus.")
    parser.add_argument("--limit", type=int, default=5, help="Retrieved chunks to use (default: 5).")
    parser.add_argument("--max-context-chars", type=int, default=18000,
                        help="Maximum total evidence characters sent to the model.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL,
                        help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL}).")
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTOR_INDEX)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_BM25_INDEX)
    parser.add_argument("--candidates", type=int, default=50,
                        help="Candidate pool for hybrid retrieval (default: 50).")
    parser.add_argument("--semantic-weight", type=float, default=3.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    args = parser.parse_args()

    if args.limit < 1 or args.candidates < args.limit:
        parser.error("--limit must be positive and --candidates must be >= --limit")
    if args.max_context_chars < 1:
        parser.error("--max-context-chars must be positive")

    try:
        vectors, metadata = load_vector_index(args.vectors, args.metadata)
        bm25_index = json.loads(args.bm25_index.read_text(encoding="utf-8"))
        results = hybrid_search(
            args.question,
            vectors,
            metadata,
            bm25_index,
            limit=args.limit,
            candidate_limit=args.candidates,
            semantic_weight=args.semantic_weight,
            bm25_weight=args.bm25_weight,
        )
        if not results:
            print("No se recuperó evidencia para responder la pregunta.")
            return 0
        context, sources = build_context(results, args.max_context_chars)
        if not sources:
            print("Los fragmentos recuperados no contienen texto utilizable.")
            return 0
        answer = ask_ollama(args.question, context, args.ollama_url, args.model)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    allowed_ids = {source["id"] for source in sources}
    cited_ids = set(CITATION_PATTERN.findall(answer))
    unknown_ids = sorted(cited_ids - {source_id[1:] for source_id in allowed_ids}, key=int)
    if unknown_ids:
        print("Advertencia: el modelo usó citas que no corresponden a las fuentes enviadas: "
              + ", ".join(f"[S{source_id}]" for source_id in unknown_ids), file=sys.stderr)
    if not cited_ids:
        print("Advertencia: la respuesta no incluyó identificadores de cita.", file=sys.stderr)

    print(answer or "El modelo no generó una respuesta.")
    print("\nFuentes recuperadas:")
    for source in sources:
        pages = ", ".join(map(str, source["pages"]))
        print(f"[{source['id']}] {source['title']} — p. {pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
