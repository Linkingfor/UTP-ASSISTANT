"""
UTP Assistant - interfaz en Streamlit.

Recibe un correo de cliente, ejecuta el ciclo del Run (análisis, requires_action,
ejecución de funciones, submit_tool_outputs, completed) y muestra el resumen para
el equipo interno. También permite conversar con el asistente dentro del mismo hilo
y revisar los sistemas simulados (Jira, Google Calendar y CRM).
"""
import json
from pathlib import Path

import streamlit as st

from asistente import (
    EQUIPO_CUENTA,
    MODELO,
    NOMBRE_PROVEEDOR,
    _leer,
    crear_cliente,
    ejecutar_run,
    formatear_correo,
    reiniciar_sistemas,
)

st.set_page_config(page_title="UTP Assistant", page_icon="📨", layout="wide")

CARPETA = Path(__file__).parent
CORREO_EJEMPLO = (
    "Hola equipo de UTP Consult, gracias por la propuesta. Nos interesa avanzar. ¿Podríamos tener una "
    "reunión la próxima semana para discutir los detalles técnicos del módulo de pagos? Adjunto un "
    "documento con algunos requisitos iniciales.\n\nSaludos,\nAna Torres de TechCorp."
)
ADJUNTO_EJEMPLO = CARPETA / "ejemplos" / "requisitos_iniciales.txt"
ICONOS = {"queued": "🕓", "in_progress": "⚙️", "requires_action": "🛠️", "submit_tool_outputs": "📤",
          "completed": "✅", "incomplete": "⚠️"}

# ---------- Estado de la sesión: el hilo (Thread) y el registro de Runs ----------
if "hilo" not in st.session_state:
    st.session_state.hilo = []
    st.session_state.runs = []


def nuevo_hilo():
    st.session_state.hilo = []
    st.session_state.runs = []


def leer_adjunto(archivo):
    if archivo is None:
        return None, None
    if archivo.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        texto = "\n".join((p.extract_text() or "") for p in PdfReader(archivo).pages)
    else:
        texto = archivo.getvalue().decode("utf-8", errors="ignore")
    return archivo.name, texto


def aplanar(filas):
    """Convierte listas y diccionarios anidados en texto para mostrarlos en una tabla."""
    planas = []
    for fila in filas:
        plana = {}
        for clave, valor in fila.items():
            if isinstance(valor, list):
                plana[clave] = ", ".join(v["nota"] if isinstance(v, dict) and "nota" in v else str(v) for v in valor)
            else:
                plana[clave] = valor
        planas.append(plana)
    return planas


def mostrar_respuesta(texto):
    """Los resúmenes de siete secciones van en texto plano; las respuestas de chat, en Markdown."""
    if texto.lstrip().startswith("1."):
        st.text(texto)
    else:
        st.markdown(texto)


def mostrar_evento(evento):
    """Dibuja un paso del ciclo del Run."""
    estado = evento["estado"]
    icono = ICONOS.get(estado, "•")
    if estado == "requires_action":
        st.markdown(f"{icono} **requires_action**: el modelo pide {len(evento['tool_calls'])} función(es). Modelo: `{evento['modelo']}`")
        for llamada in evento["tool_calls"]:
            st.markdown(f"↳ `{llamada['function']['name']}`")
            st.json(json.loads(llamada["function"]["arguments"]), expanded=False)
    elif estado == "submit_tool_outputs":
        st.markdown(f"{icono} **submit_tool_outputs**: resultados devueltos al modelo.")
        for salida in evento["tool_outputs"]:
            st.markdown(f"↳ `{salida['funcion']}` → `{json.dumps(salida['output'], ensure_ascii=False)}`")
    else:
        st.markdown(f"{icono} **{estado}**: {evento.get('detalle', '')}")


def correr(contenido, titulo):
    """Ejecuta un Run mostrando el avance en vivo y guarda el resultado."""
    with st.status("Run en ejecución...", expanded=True) as estado:
        try:
            cliente = crear_cliente(api_key)
            texto, registro = ejecutar_run(cliente, st.session_state.hilo, contenido, al_avanzar=mostrar_evento)
            estado.update(label=f"Run finalizado: {registro[-1]['estado']}", state="complete", expanded=True)
        except Exception as error:
            estado.update(label="Run fallido", state="error")
            st.error(f"No se pudo completar el Run: {error}")
            return None
    st.session_state.runs.append({"titulo": titulo, "registro": registro, "respuesta": texto})
    return texto


# ---------- Barra lateral ----------
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("API key (opcional si está en .env)", type="password",
                            help="Clave gratuita de Gemini: https://aistudio.google.com/apikey")
    st.caption(f"Proveedor: {NOMBRE_PROVEEDOR}")
    st.caption(f"Modelo: `{MODELO}`")
    st.caption(f"Equipo de la cuenta: {', '.join(EQUIPO_CUENTA)}")
    st.button("🧵 Nuevo hilo (Thread)", on_click=nuevo_hilo)
    st.button("🗑️ Reiniciar sistemas simulados", on_click=reiniciar_sistemas)
    st.divider()
    st.caption(f"Mensajes en el hilo: {len(st.session_state.hilo)} · Runs ejecutados: {len(st.session_state.runs)}")

# ---------- Zona principal ----------
st.title("📨 UTP Assistant")
st.caption("Asistente de gestión de proyectos y ventas de UTPConsult · Tarea Académica 2 · Python + Streamlit + API compatible con OpenAI (Gemini)")

tab_correo, tab_chat, tab_sistemas, tab_runs = st.tabs(
    ["📥 Procesar correo", "💬 Conversar con el asistente", "🗂️ Sistemas simulados", "🧾 Registro de Runs"]
)

with tab_correo:
    col_izq, col_der = st.columns([1, 1])
    with col_izq:
        st.subheader("Correo entrante")
        remitente = st.text_input("De", "Ana Torres <ana.torres@techcorp.com>")
        asunto = st.text_input("Asunto", "Re: Propuesta para el módulo de pagos")
        cuerpo = st.text_area("Cuerpo del correo", CORREO_EJEMPLO, height=200)
        archivo = st.file_uploader("Adjunto (txt, md o pdf)", type=["txt", "md", "pdf"])
        usar_ejemplo = st.checkbox("Usar el adjunto de ejemplo (requisitos iniciales de TechCorp)", value=archivo is None)
        procesar = st.button("▶️ Procesar correo (crear Run)", type="primary")
    with col_der:
        st.subheader("Ciclo del Run y respuesta para el equipo")
        if procesar:
            nombre_adjunto, texto_adjunto = leer_adjunto(archivo)
            if texto_adjunto is None and usar_ejemplo and ADJUNTO_EJEMPLO.exists():
                nombre_adjunto, texto_adjunto = ADJUNTO_EJEMPLO.name, ADJUNTO_EJEMPLO.read_text(encoding="utf-8")
            contenido = formatear_correo(remitente, asunto, cuerpo, nombre_adjunto, texto_adjunto)
            respuesta = correr(contenido, f"Correo: {asunto}")
            if respuesta:
                with st.container(border=True):
                    mostrar_respuesta(respuesta)
        elif st.session_state.runs:
            ultimo = st.session_state.runs[-1]
            st.caption(f"Último Run: {ultimo['titulo']}")
            with st.container(border=True):
                mostrar_respuesta(ultimo["respuesta"])
        else:
            st.info("Escribe o pega un correo y pulsa **Procesar correo**. Aquí verás cada estado del Run y el resumen final.")

with tab_chat:
    st.subheader("Conversación con el asistente dentro del hilo")
    for mensaje in st.session_state.hilo:
        contenido = mensaje.get("content") or ""
        if mensaje["role"] == "user":
            with st.chat_message("user"):
                if contenido.startswith("CORREO ENTRANTE"):
                    with st.expander("📧 Correo procesado (clic para ver)"):
                        st.text(contenido)
                else:
                    st.markdown(contenido)
        elif mensaje["role"] == "assistant" and contenido:
            with st.chat_message("assistant"):
                if contenido.lstrip().startswith("1."):
                    with st.expander("📋 Resumen del Run para el equipo (clic para ver)"):
                        st.text(contenido)
                else:
                    st.markdown(contenido)
    pregunta = st.chat_input("Pregunta o instrucción para el asistente (por ejemplo: ¿qué queda pendiente con TechCorp?)")
    if pregunta:
        with st.chat_message("user"):
            st.markdown(pregunta)
        with st.chat_message("assistant"):
            respuesta = correr(pregunta, f"Instrucción: {pregunta[:40]}")
            if respuesta:
                mostrar_respuesta(respuesta)

with tab_sistemas:
    c1, c2, c3 = st.columns(3)
    tickets, eventos, contactos = _leer("jira"), _leer("calendario"), _leer("crm")
    c1.metric("Tickets en Jira", len(tickets))
    c2.metric("Eventos en el calendario", len(eventos))
    c3.metric("Contactos en el CRM", len(contactos))
    for titulo, filas in (("Jira (simulado)", tickets), ("Google Calendar (simulado)", eventos), ("CRM (simulado)", contactos)):
        st.subheader(titulo)
        if filas:
            st.dataframe(aplanar(filas), width="stretch")
        else:
            st.caption("Sin registros todavía.")

with tab_runs:
    if not st.session_state.runs:
        st.caption("Todavía no se ejecutó ningún Run en este hilo.")
    for numero, run in enumerate(st.session_state.runs, start=1):
        with st.expander(f"Run {numero}: {run['titulo']} · estado final: {run['registro'][-1]['estado']}"):
            for evento in run["registro"]:
                mostrar_evento(evento)
