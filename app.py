"""
Interfaz conversacional con Streamlit para el sistema RAG multimodal.
Proyecto Final - Recuperación de Información
Ejecutar: streamlit run app.py
"""

import streamlit as st
import os
from rag_system import ask
from retrieve import search  # disponible por si se usa directamente
from feedback_store import add_feedback, get_stats


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA (título, icono, diseño ancho)
# ============================================================
st.set_page_config(
    page_title="Sistema RI Multimodal",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# ESTILOS CSS PERSONALIZADOS (para las tarjetas y etiquetas)
# ============================================================
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .sub-header { opacity: 0.65; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .evidence-card {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 10px;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
        background-color: var(--secondary-background-color);
    }
    .score-badge {
        background-color: rgba(99, 102, 241, 0.18);
        color: #818cf8;
        font-weight: 600;
        font-size: 0.78rem;
        padding: 2px 10px;
        border-radius: 999px;
        margin-top: 4px;
    }
    .rank-badge {
        background-color: var(--primary-color);
        color: white;
        font-weight: 700;
        font-size: 0.75rem;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        text-align: center;
        line-height: 22px;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# BARRA LATERAL (SIDEBAR): parámetros y controles del sistema
# ============================================================
with st.sidebar:
    st.markdown("### 🔎 Sistema RI Multimodal")
    st.caption("Recuperación de Información + RAG sobre texto e imágenes")
    st.markdown("---")

    st.markdown("#### ⚙️ Parámetros de búsqueda")
    top_k = st.slider("Documentos a recuperar (Top-k)", min_value=1, max_value=20, value=10)

    query_type = st.radio(
        "Tipo de consulta",
        options=["text", "image"],
        format_func=lambda x: "📝 Texto" if x == "text" else "🖼️ Imagen",
        horizontal=True,
        help="Con 'Imagen' podrás subir una foto en vez de escribir.",
    )

    use_rerank = st.toggle(
        "🎯 Re-ranking (Cross-Encoder)",
        value=True,
        help="Refina el orden de la búsqueda vectorial con un modelo más preciso.",
    )

    use_expansion = st.toggle(
        "🔀 Query Expansion",
        value=False,
        help="Genera sinónimos con LLM para mejorar el recall en búsquedas de texto.",
    )

    st.markdown("---")
    st.markdown("#### ℹ️ Acerca del sistema")
    st.markdown("""
        Combina: embeddings CLIP, base vectorial, RAG, re-ranking, 
        expansión de consulta, memoria conversacional y feedback.
    """)

    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# ENCABEZADO PRINCIPAL DE LA PÁGINA
# ============================================================
st.markdown('<p class="main-header">Sistema de Recuperación de Información Multimodal</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Consulta productos en lenguaje natural — el sistema recupera evidencia '
    'relevante (texto + imágenes) y genera una respuesta usando ese contexto.</p>',
    unsafe_allow_html=True,
)


# ============================================================
# FUNCIÓN: RENDERIZAR EVIDENCIAS (tarjetas de productos)
# ============================================================
def render_evidences(docs, msg_key):
    """Dibuja las tarjetas de los productos recuperados con sus imágenes,
    puntuaciones y botones de feedback (👍/👎). 'msg_key' asegura claves únicas."""
    if not docs:
        return

    with st.expander(f"📦 Ver evidencia utilizada ({len(docs)} productos)", expanded=False):
        for i, doc in enumerate(docs):
            pid = doc.get("product_id", f"idx{i}")
            with st.container():
                st.markdown('<div class="evidence-card">', unsafe_allow_html=True)

                col1, col2, col3 = st.columns([1, 3, 1])

                # Columna 1: Imagen del producto
                with col1:
                    img_path = doc.get("image_path", "")
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        st.markdown("<div style='text-align:center; padding-top:20px;'>🖼️<br><small>Sin imagen</small></div>", unsafe_allow_html=True)

                # Columna 2: Título, descripción y puntuaciones
                with col2:
                    st.markdown(
                        f'<span class="rank-badge">{i+1}</span>'
                        f'**{doc.get("titulo", "Sin título")}**',
                        unsafe_allow_html=True,
                    )
                    score = doc.get("score_final", 0.0)
                    badges = f'<span class="score-badge">vector: {score:.4f}</span>'
                    if "rerank_score" in doc:
                        badges += f' <span class="score-badge">rerank: {doc["rerank_score"]:.4f}</span>'
                    st.markdown(badges, unsafe_allow_html=True)

                    descripcion = doc.get("descripcion", "")
                    if descripcion and str(descripcion).lower() != "nan":
                        st.write(str(descripcion)[:180] + "...")

                # Columna 3: Feedback del usuario (likes/dislikes)
                with col3:
                    stats = get_stats(pid)
                    btn_col1, btn_col2 = st.columns(2)

                    with btn_col1:
                        if st.button("👍", key=f"like_{msg_key}_{pid}", help="Útil"):
                            add_feedback(pid, liked=True)
                            st.toast(f"Gracias por tu feedback sobre '{doc.get('titulo', '')[:30]}'")
                            st.rerun()

                    with btn_col2:
                        if st.button("👎", key=f"dislike_{msg_key}_{pid}", help="No útil"):
                            add_feedback(pid, liked=False)
                            st.toast("Feedback registrado, ajustaremos futuras búsquedas")
                            st.rerun()

                    st.caption(f"👍{stats['likes']} · 👎{stats['dislikes']}")

                st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# INICIALIZACIÓN Y RENDERIZADO DEL HISTORIAL DEL CHAT
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("user_image"):
            st.image(msg["user_image"], width=180)
        st.markdown(msg["content"])
        if msg.get("evidences"):
            render_evidences(msg["evidences"], msg_key=f"hist{msg_idx}")


# ============================================================
# INPUT DEL USUARIO (según tipo: texto o imagen)
# ============================================================
prompt = None
image_query_path = None
display_content = None

if query_type == "text":
    prompt = st.chat_input("Escribe tu consulta sobre productos...")
    if prompt:
        display_content = prompt
else:
    uploaded_file = st.file_uploader(
        "Sube una imagen de referencia",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded_file is not None:
        st.image(uploaded_file, width=180, caption="Imagen a buscar")
        if st.button("🔍 Buscar productos similares", type="primary"):
            os.makedirs("data/tmp_uploads", exist_ok=True)
            temp_path = os.path.join(
                "data/tmp_uploads",
                f"{len(st.session_state.messages)}_{uploaded_file.name}"
            )
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_query_path = temp_path
            display_content = f"🖼️ Imagen subida: {uploaded_file.name}"


# ============================================================
# FUNCIÓN AUXILIAR: CONSTRUIR MEMORIA DE LA CONVERSACIÓN
# ============================================================
def build_memoria_context(messages, max_turns=3):
    """Devuelve un texto con los últimos intercambios (usuario/asistente)
    para usarlo como memoria en la generación de la respuesta."""
    if not messages:
        return ""
    memo = []
    relevant_msgs = messages[-(max_turns * 2):]
    for m in relevant_msgs:
        if m["role"] == "user":
            memo.append(f"Usuario: {m['content']}")
        elif m["role"] == "assistant":
            memo.append(f"Asistente: {m['content']}")
    return "\n".join(memo)


# ============================================================
# PROCESAMIENTO DE LA CONSULTA Y GENERACIÓN DE RESPUESTA
# ============================================================
if prompt or image_query_path:

    # Capturamos la memoria ANTES de añadir el mensaje actual
    memoria_context = build_memoria_context(st.session_state.messages, max_turns=3)
    actual_query = prompt if query_type == "text" else image_query_path

    # Guardamos y mostramos el mensaje del usuario
    st.session_state.messages.append({
        "role": "user",
        "content": display_content,
        "user_image": image_query_path if query_type == "image" else None,
    })
    with st.chat_message("user"):
        if image_query_path:
            st.image(image_query_path, width=180)
        st.markdown(display_content)

    # Generamos la respuesta del asistente
    with st.chat_message("assistant"):
        with st.spinner("Recuperando evidencia y generando respuesta..."):
            try:
                # Llamada principal al sistema RAG
                response, docs = ask(
                    actual_query,
                    query_type=query_type,
                    top_k=top_k,
                    memory=memoria_context,
                    use_rerank=use_rerank,
                    use_expansion=use_expansion,
                )
            except Exception as e:
                response = f"⚠️ Ocurrió un error: {e}"
                docs = []

        # Guardamos la respuesta (importante hacerlo antes de renderizar feedbacks)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "evidences": docs,
        })
        new_msg_idx = len(st.session_state.messages) - 1

        # Mostramos respuesta y evidencias
        st.markdown(response)
        render_evidences(docs, msg_key=f"hist{new_msg_idx}")