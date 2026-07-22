# Sistema de Recuperación de Información Multimodal con RAG
 
Proyecto final — Asignatura: Recuperación de Información

Integrantes

- Freddy Jiménez
- Erick Romero
 
Sistema de recuperación de información multimodal (texto + imágenes) sobre un corpus de productos, con generación de respuestas mediante RAG (Retrieval-Augmented Generation), interfaz conversacional en Streamlit, y funcionalidades de excelencia: **Relevance Feedback**, **Memoria conversacional**, **Re-ranking** y **Query Expansion**.
 
## Índice
 
- [Arquitectura del sistema](#arquitectura-del-sistema)
- [Requisitos previos](#requisitos-previos)
- [Instalación](#instalación)
- [Configuración (.env)](#configuración-env)
- [Preparar los datos (una sola vez)](#preparar-los-datos-una-sola-vez)
- [Ejecutar la aplicación](#ejecutar-la-aplicación)
- [Evaluación del sistema](#evaluación-del-sistema)
- [Funcionalidades de excelencia](#funcionalidades-de-excelencia)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Solución de problemas comunes](#solución-de-problemas-comunes)
## Arquitectura del sistema
 
```
Corpus (texto + imágenes)
        │
        ▼
generate_embeddings.py  →  Embeddings CLIP (ViT-B-32) por producto
        │
        ▼
vector_db.py  →  Indexación en ChromaDB (colecciones de texto e imagen)
        │
        ▼
retrieve.py  →  Búsqueda vectorial + Query Expansion + Relevance Feedback + Re-ranking
        │
        ▼
rag_system.py  →  Generación de respuesta con LLM (OpenRouter) usando el contexto recuperado
        │
        ▼
app.py  →  Interfaz conversacional (Streamlit)
```
 
**Modelos usados:**
- **CLIP** (`ViT-B-32`, pesos `laion2b_s34b_b79k` vía `open_clip`) para embeddings multimodales
- **ChromaDB** como base de datos vectorial
- **Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) para re-ranking
- **LLM vía OpenRouter** (modelos gratuitos con soporte de visión) para la generación de respuestas y la expansión de consultas
## Requisitos previos
 
- Python 3.11
- GPU NVIDIA con CUDA (recomendado; el sistema también corre en CPU pero más lento)
- Cuenta gratuita en [OpenRouter](https://openrouter.ai) para obtener una API key
## Instalación
 
### 1. Clonar el repositorio
 
```bash
git clone https://github.com/Erickezro/Sistema-de-RI-Multimodal-con-RAG.git
cd Sistema-de-RI-Multimodal-con-RAG
```
 
### 2. Crear y activar un entorno virtual
 
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows (PowerShell)
# source venv/bin/activate           # Linux / macOS
```
 
### 3. Instalar PyTorch con soporte CUDA
 
Instalar **antes** que el resto de dependencias, apuntando al índice específico de PyTorch:
 
```powershell
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```
 
> Si no tienes GPU o prefieres CPU, omite el `--index-url` y usa `pip install torch torchvision torchaudio` en su lugar (la ejecución será más lenta, especialmente en `generate_embeddings.py`).
 
### 4. Instalar el resto de dependencias
 
```powershell
pip install -r requirements.txt
```
 
### 5. Verificar que CUDA quedó disponible (opcional)
 
```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
 
## Configuración (.env)
 
Crea un archivo `.env` en la raíz del proyecto con tu API key de OpenRouter:
 
```
OPENROUTER_API_KEY=tu_api_key_aqui
```
 
Puedes obtener una key gratuita en [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys).
 
> El sistema usa modelos **gratuitos** de OpenRouter con soporte de visión. Estos modelos a veces se saturan temporalmente del lado del proveedor (error 429); el código ya incluye una lista de modelos de respaldo que se prueban en cadena automáticamente para mitigar esto.
 
## Preparar los datos (una sola vez)
 
Ejecuta estos scripts **en orden** la primera vez que configures el proyecto:
 
```powershell
# 1. Descargar el corpus
python download_corpus.py
 
# 2. Limpiar y estructurar los datos
python clean_corpus.py
 
# 3. Generar embeddings CLIP para todos los productos
python generate_embeddings.py
 
# 4. Indexar los embeddings en ChromaDB
python vector_db.py
```
 
> `generate_embeddings.py` puede tardar varios minutos dependiendo de si corre en GPU o CPU. Al finalizar, deberías ver confirmación de que se generaron los embeddings de texto e imagen para todos los productos.
 
## Ejecutar la aplicación
 
```powershell
streamlit run app.py
```
 
Esto abre la interfaz en `http://localhost:8501`. Desde ahí puedes:
 
- Escribir consultas en lenguaje natural, o subir una imagen de referencia (toggle **Texto / Imagen** en el sidebar)
- Ajustar el número de resultados (**Top-k**)
- Activar/desactivar **Re-ranking** y **Query Expansion**
- Calificar los resultados con 👍 / 👎 (Relevance Feedback)
- Ver la evidencia usada (productos, imágenes y scores) en cada respuesta
## Evaluación del sistema
 
Para calcular Precision@k, Recall@k y NDCG@k sobre el conjunto de qrels:
 
```powershell
python evaluate.py
```
 
Esto imprime una tabla con las métricas para k = 1, 3, 5, 10. La evaluación usa `retrieve.search()` directamente (sin pasar por el LLM), por lo que corre en segundos y es totalmente reproducible.
 
## Funcionalidades de excelencia
 
| Funcionalidad | Archivo(s) | Descripción |
|---|---|---|
| **Relevance Feedback** | `feedback_store.py` | Los votos 👍/👎 se guardan en `data/feedback.json` y ajustan el score de futuras búsquedas para ese producto. |
| **Memoria conversacional** | `app.py` (`build_memoria_context`) | Los últimos turnos de la conversación se pasan al LLM al generar la respuesta (no afectan la búsqueda vectorial, para no degradar el embedding de la consulta actual). |
| **Re-ranking** | `rerank.py` | Un Cross-Encoder reordena un pool ampliado de candidatos (bi-encoder CLIP → cross-encoder) antes de devolver el Top-k final. Solo aplica a consultas de texto. |
| **Query Expansion** | `query_expansion.py` | Un LLM genera reformulaciones/sinónimos de la consulta; se busca con todas las variantes y se combinan resultados por el score máximo obtenido entre ellas. Se degrada de forma segura a la consulta original si el LLM falla. |
 
Todas son activables/desactivables desde el sidebar de la app (excepto Relevance Feedback, que siempre está disponible vía los botones 👍/👎).
 
## Estructura del proyecto
 
```
├── app.py                  # Interfaz Streamlit
├── clean_corpus.py                  # Limpieza del corpus
├── download_corpus.py                  # Descarga del corpus
├── rag_system.py            # Pipeline RAG (búsqueda + generación con LLM)
├── retrieve.py               # Búsqueda vectorial (texto/imagen, feedback, rerank, expansion)
├── rerank.py                 # Re-ranking con Cross-Encoder
├── query_expansion.py        # Generación de reformulaciones de consulta
├── feedback_store.py         # Persistencia de relevance feedback
├── generate_embeddings.py    # Generación de embeddings CLIP del corpus
├── vector_db.py              # Indexación en ChromaDB
├── evaluate.py                # Evaluación (Precision@k, Recall@k, NDCG@k)
├── requirements.txt
├── .env                       # API key de OpenRouter (no versionado)
└── data/
    ├── raw/                   # Corpus original (imágenes, etc.)
    ├── processed/             # products.csv, qrels.csv, embeddings (.npy)
    ├── chroma_db/              # Base de datos vectorial persistida
    ├── feedback.json           # Relevance feedback acumulado
    └── tmp_uploads/             # Imágenes subidas temporalmente por el usuario
```
 
## Solución de problemas comunes
 
**Error `ModuleNotFoundError`** al correr un script: confirma que el entorno virtual está activado (`(venv)` visible en la terminal) y que corriste `pip install -r requirements.txt`.
 
**Access violation / crash sin traceback al leer el CSV**: puede deberse a una instalación corrupta de `pyarrow`. Solución:
```powershell
pip uninstall pyarrow -y
pip install pyarrow --force-reinstall
```
 
**Error 404/429 al generar respuesta con el LLM**: los modelos gratuitos de OpenRouter cambian de disponibilidad con frecuencia y a veces se saturan temporalmente. El sistema ya reintenta con una lista de modelos de respaldo automáticamente; si el error persiste, revisa la lista `FALLBACK_MODELS` en `rag_system.py` y `EXPANSION_MODELS` en `query_expansion.py` contra los modelos `:free` vigentes en [openrouter.ai/models](https://openrouter.ai/models?max_price=0).
 
**`torch.cuda.is_available()` devuelve `False`**: confirma que instalaste la versión de PyTorch con el índice `cu128` (paso 3 de instalación), no la versión CPU-only, y que tienes el driver de NVIDIA actualizado.