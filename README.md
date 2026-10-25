# RAG Project

A Retrieval-Augmented Generation project built with Python.

## Búsqueda léxica (BM25)

Genera primero un JSON de fragmentos con `src/ingestion/chunk_text.py` y guárdalo en una ruta local ignorada por Git, por ejemplo `data/chunks/documento.json`. Después, desde la raíz del proyecto:

```bash
python src/retrieval/bm25.py build data/chunks/*.json
python src/retrieval/bm25.py search "MRtrix tractography" --limit 5
```

El índice predeterminado queda en `data/index/bm25.json`. Contiene texto extraído, por lo que debe permanecer en una ruta local ignorada por Git. La búsqueda muestra los fragmentos mejor puntuados con su título y página; no genera respuestas.
