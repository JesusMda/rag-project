# Registro de problemas y soluciones

Este archivo es la bitácora del proyecto. Añadiremos cada hallazgo con su efecto, solución y estado para que las decisiones de procesamiento del corpus sean rastreables.

## 2026-11-14 — Revisión de extracción y fragmentación

### Caracteres de control en la tesis

- **Problema:** la extracción de la tesis *Partículas Masivas para Tractografía Cerebral usando Multi-Tensores de Difusión* contenía caracteres de control (por ejemplo, `\x01`) que ensuciaban el texto de sus fragmentos.
- **Solución:** el extractor ahora convierte caracteres de control en espacios, conserva tabuladores y saltos de línea, y elimina caracteres de formato invisibles. No altera el PDF original.
- **Estado:** corregido en el extractor y en la limpieza de fragmentos. Se regeneró la extracción de la tesis y sus fragmentos locales.

### Continuidad entre fragmentos

- **Problema:** un fragmento puede terminar a mitad de una explicación al alcanzar el límite de tamaño.
- **Solución:** se conserva el solapamiento configurado de 150 caracteres para incluir el contexto inmediato en el fragmento siguiente. No se cambiaron los límites porque la fragmentación actual ya está indexada; cambiarla obliga a recalcular todos los embeddings. Si la evaluación detecta fallos de recuperación por cortes, probar límites por oración y comparar antes de adoptar el cambio.
- **Estado:** mitigado por el solapamiento; pendiente de validar con preguntas de evaluación.

### Páginas citadas frente a numeración impresa

- **Problema:** en libros y artículos, la página impresa puede diferir de la posición de la página dentro del PDF, especialmente por portadas y páginas preliminares.
- **Solución:** la extracción conserva las páginas físicas del PDF, numeradas desde 1. El README aclara que las citas usan esta numeración. Las páginas vacías se conservan en la extracción para no desplazar las referencias y no generan fragmentos.
- **Estado:** documentado. Al verificar una cita, buscar ese número de página del visor PDF; la numeración impresa puede ser distinta.

### Texto defectuoso por el diseño del PDF

- **Problema:** además de los caracteres de control, algunas páginas de la tesis tienen palabras con letras espaciadas o palabras concatenadas. Cambiar el orden de extracción no produce una mejora consistente y una limpieza automática podría alterar fórmulas o términos.
- **Solución:** se eliminan los caracteres de control conocidos; no se intenta insertar espacios mediante heurísticas agresivas.
- **Comprobación:** se comparó una página con extracción `plain` y `layout` de pypdf y Poppler. Poppler redujo algunos patrones de letras espaciadas en esa muestra, pero no los eliminó y produjo un carácter de control de página; no es suficiente para cambiar el extractor de todo el corpus. Tesseract no está instalado en este entorno.
- **Estado:** los controles quedaron corregidos y los fragmentos se regeneraron. Las palabras concatenadas o con espaciado anómalo siguen pendientes. Hace falta OCR explícito en las páginas afectadas, con revisión visual contra el PDF antes de incorporarlo.

## Resultado de esta revisión

- Se regeneraron los 25 archivos de fragmentos (5.209 fragmentos) y el índice BM25.
- La limpieza modificó un único fragmento de la tesis; se recalculó su vector semántico y se conservaron los vectores idénticos del resto del corpus.
- Los índices BM25 y semántico y los archivos de fragmentos coinciden: 5.209 documentos/vectores.

## Plantilla para nuevas entradas

### AAAA-MM-DD — Título breve

- **Problema:** qué se observó y dónde.
- **Solución:** cambio realizado o decisión tomada.
- **Estado:** corregido, en revisión o pendiente.

## 2026-11-14 — Respuesta de Ollama sin límite de longitud

- **Problema:** al validar preguntas, Ollama tardaba y la solicitud agotaba el tiempo de espera. La API aceptaba consultas breves.
- **Solución:** el comando de respuestas ahora envía `num_predict` y ofrece `--max-answer-tokens` (256 por defecto) para acotar la generación.
- **Estado:** implementado. `llama3.1` terminó con error HTTP 500 (proceso del modelo terminado por el servidor); la validación usó `llama3.2:1b`.

## 2026-11-14 — Respuestas sin citas o fuera de evidencia

- **Problema:** `llama3.2:1b` omitió citas en las tres pruebas. En la pregunta sobre NODDI confundió cuatro *shells* con capas de tejido; en la pregunta ambigua sobre el “mejor método” afirmó que había uno universal; ante “¿Cuál es la capital de Francia?” contestó con conocimiento externo y fragmentos irrelevantes.
- **Solución:** el CLI ahora reemplaza la respuesta por un aviso si no hay citas o si alguna cita no pertenece a las fuentes enviadas. El README explica esta protección.
- **Estado:** protección implementada. Las pruebas revelan que este modelo no cumple el formato solicitado, así que la etapa de respuestas no se puede dar por validada. Incluso con identificadores válidos, hay que comprobar cada afirmación contra el PDF.
- **Revisión de cita:** la fuente NODDI indicada en la prueba apuntaba al PDF, página 5, donde se describen experimentos sintéticos de precisión y comparación contra el *ground truth*. La respuesta del modelo añadió una afirmación errónea sobre “cuatro capas”.
