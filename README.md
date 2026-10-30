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

## Respuestas fundamentadas (Ollama local)

La generación usa Ollama en `http://localhost:11434` para mantener local el procesamiento de los documentos. Instala/inicia Ollama y descarga un modelo compatible, por ejemplo `ollama pull llama3.1`. Después de construir los índices léxico y semántico, pregunta así:

```bash
python src/retrieval/answering.py "¿Qué mide NODDI y cómo se evalúa su precisión?"
python src/retrieval/answering.py "¿Qué sesgos afectan a la tractografía?" --model llama3.1 --limit 8
```

El comando recupera fragmentos con búsqueda híbrida, envía al modelo solo esos fragmentos y solicita citas `[S1]`, `[S2]` enlazadas con título y página. `--max-context-chars` limita el tamaño de la evidencia enviada. Se puede seleccionar otra instancia compatible con la API de Ollama usando `--ollama-url`; esa instancia recibirá la pregunta y el texto recuperado, así que usa una instancia de confianza. La generación puede omitir o asignar mal citas; comprueba cada referencia contra el PDF original.
