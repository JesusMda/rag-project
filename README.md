# RAG Project

A Retrieval-Augmented Generation project built with Python.

## Búsqueda léxica (BM25)

Genera primero un JSON de fragmentos con `src/ingestion/chunk_text.py` y guárdalo en una ruta local ignorada por Git, por ejemplo `data/chunks/documento.json`. Después, desde la raíz del proyecto:

```bash
python src/retrieval/bm25.py build data/chunks/*.json
python src/retrieval/bm25.py search "MRtrix tractography" --limit 5
```

El índice predeterminado queda en `data/index/bm25.json`. Contiene texto extraído, por lo que debe permanecer en una ruta local ignorada por Git. La búsqueda muestra los fragmentos mejor puntuados con su título y página; no genera respuestas.

## Búsqueda semántica e híbrida

Instala las dependencias con `pip install -r requirements.txt`. La primera construcción descarga el modelo multilingüe `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; después, el modelo y los índices se usan localmente. Genera los vectores una vez y compara las mismas preguntas con búsqueda semántica o BM25 más semántica:

```bash
python src/retrieval/semantic.py build data/chunks/*.json
python src/retrieval/semantic.py search "¿Qué mide NODDI?" --limit 5
python src/retrieval/semantic.py hybrid "¿Qué mide NODDI?" --limit 5
```

Los vectores y metadatos se guardan en `data/index/semantic.npz` y `data/index/semantic.json`, ambos locales e ignorados por Git. La combinación híbrida usa Reciprocal Rank Fusion ponderada (semántica 3:1 léxica por defecto) para combinar posiciones sin comparar directamente BM25 y similitud coseno. Los pesos se pueden ajustar con `--semantic-weight` y `--bm25-weight`.
