"""
Relevance Feedback - almacenamiento y aplicación.
 
Guarda, por producto, cuántas veces fue marcado como útil (👍) o no útil (👎).
Esa señal se usa en retrieve.py para dar un pequeño boost/penalización
a los productos en búsquedas futuras, sin necesitar re-entrenar nada.
 
Persistencia simple en JSON (suficiente para un proyecto académico;
si el corpus creciera mucho convendría moverlo a una tabla en la BD vectorial).
"""
 
import json
import logging
import os

logger = logging.getLogger(__name__)

FEEDBACK_PATH = "data/feedback.json"
 
# Cuánto pesa cada "voto neto" en el score final, y el tope para que
# el feedback no termine dominando por completo la similitud semántica.
ALPHA = 0.03
CAP = 0.15
 
 
def _ensure_dir():
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
 
 
def load_feedback() -> dict:
    if not os.path.exists(FEEDBACK_PATH):
        return {}
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        # El feedback es una señal auxiliar (no crítica): si el archivo
        # está corrupto o ilegible NO tumbamos toda la app/búsqueda por
        # eso. Degradamos a "sin feedback" pero dejamos rastro del error.
        logger.warning("No se pudo leer %s (%s); se ignora el feedback acumulado.",
                       FEEDBACK_PATH, e)
        return {}
    if not isinstance(data, dict):
        logger.warning("%s no contiene un objeto JSON válido; se ignora.", FEEDBACK_PATH)
        return {}
    return data
 
 
def save_feedback(data: dict):
    _ensure_dir()
    # Escritura atómica: escribimos a un archivo temporal y luego lo
    # renombramos. Así una escritura interrumpida (ej. un rerun de
    # Streamlit a mitad de guardado) nunca deja el JSON corrupto.
    tmp_path = f"{FEEDBACK_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, FEEDBACK_PATH)
 
 
def add_feedback(product_id: str, liked: bool):
    """Registra un 👍 (liked=True) o 👎 (liked=False) para un producto."""
    data = load_feedback()
    entry = data.get(product_id) or {}
    likes = int(entry.get("likes", 0))
    dislikes = int(entry.get("dislikes", 0))
    if liked:
        likes += 1
    else:
        dislikes += 1
    data[product_id] = {"likes": likes, "dislikes": dislikes}
    save_feedback(data)
 
 
def get_boost(product_id: str, data: dict | None = None) -> float:
    """
    Devuelve un ajuste de score en [-CAP, +CAP] según el feedback acumulado
    del producto. Se puede pasar `data` ya cargado para evitar leer el
    JSON en cada producto dentro de un loop de búsqueda.
    """
    if data is None:
        data = load_feedback()
    entry = data.get(product_id)
    if not entry:
        return 0.0
    net = int(entry.get("likes", 0)) - int(entry.get("dislikes", 0))
    boost = ALPHA * net
    return max(-CAP, min(CAP, boost))
 
 
def get_stats(product_id: str) -> dict:
    data = load_feedback()
    return data.get(product_id, {"likes": 0, "dislikes": 0})