"""
UTP Assistant - interfaz en Streamlit.

Recibe un correo de cliente, ubica el hilo (Thread) de ese cliente, ejecuta el ciclo
del Run (análisis, requires_action, ejecución de funciones, submit_tool_outputs,
completed) y muestra el resumen para el equipo interno. También permite conversar
con el asistente dentro del hilo, aprobar o rechazar las reuniones tentativas y revisar
los sistemas simulados (Jira, Google Calendar y CRM) y la auditoría.
"""
import json
import re
from pathlib import Path

import streamlit as st

from asistente import (
    ACCIONES_CON_APROBACION,
    EQUIPO_CUENTA,
    MAXIMO_MENSAJES,
    MODELO,
    NOMBRE_PROVEEDOR,
    aprobar_evento,
    cargar_hilos,
    crear_cliente,
    ejecutar_run,
    estado_sistemas,
    formatear_correo,
    formatear_fecha,
    guardar_hilo,
    leer_auditoria,
    obtener_hilo_id,
    plural,
    preparar_demostracion,
    registrar_auditoria,
)

st.set_page_config(page_title="UTP Assistant", page_icon="📨", layout="wide")

CARPETA = Path(__file__).parent
ESCENARIOS = json.loads((CARPETA / "ejemplos" / "escenarios.json").read_text(encoding="utf-8"))
ADJUNTO_EJEMPLO = CARPETA / "ejemplos" / "requisitos_iniciales.txt"
PREFIJO_EQUIPO = "MENSAJE DEL EQUIPO INTERNO: "
INSIGNIAS = {
    "queued": ":gray-badge[queued]", "in_progress": ":blue-badge[in_progress]",
    "requires_action": ":orange-badge[requires_action]", "submit_tool_outputs": ":violet-badge[submit_tool_outputs]",
    "completed": ":green-badge[completed]", "incomplete": ":red-badge[incomplete]", "failed": ":red-badge[failed]",
    "cancelled": ":red-badge[cancelled]",
}
LEYENDA = [
    ("queued", "El Run se creó y espera turno en el hilo del cliente."),
    ("in_progress", "El modelo lee el hilo y decide qué hacer."),
    ("requires_action", "El modelo pide una o más funciones (tool_calls) con sus argumentos."),
    ("submit_tool_outputs", "El sistema valida, ejecuta las funciones y devuelve los resultados al modelo."),
    ("completed", "El modelo redacta la respuesta final para el equipo."),
    ("incomplete", "La respuesta se recortó por el límite de tokens, el modelo no devolvió texto o el Run agotó sus ciclos; se guarda lo que alcanzó a redactar."),
    ("failed", "Falló el proveedor (cuota agotada, sin conexión) o hubo un error inesperado; el hilo vuelve a su estado previo y el error queda en la auditoría."),
    ("cancelled", "El Run se interrumpió por una acción del usuario; lo ya ejecutado quedó anotado en el hilo."),
]
ORDEN_CALENDARIO = ["evento_id", "titulo", "inicio", "estado", "invitaciones_enviadas", "asistentes_cliente", "modalidad",
                    "agenda", "fin", "asistentes_internos", "enlace", "decidido_por", "fecha_decision", "hilo_id", "correo_origen_id"]

# ---------- Estado de la sesión: hilo activo, mensajes y Runs ----------
if "hilo_id" not in st.session_state:
    st.session_state.hilo_id = None
    st.session_state.hilo = []
    st.session_state.runs = {}


def activar_hilo(hilo_id):
    """Ubica el Thread del cliente (o empieza uno nuevo) y carga sus mensajes."""
    datos = cargar_hilos().get(hilo_id, {}) if hilo_id else {}
    st.session_state.hilo_id = hilo_id
    st.session_state.hilo = datos.get("mensajes", [])
    st.session_state.cliente_hilo = datos.get("cliente") or hilo_id


def aplicar_escenario():
    escenario = ESCENARIOS[st.session_state.escenario]
    st.session_state.remitente = escenario["de"]
    st.session_state.asunto = escenario["asunto"]
    st.session_state.cuerpo = escenario["cuerpo"]
    st.session_state.usar_ejemplo = escenario["adjunto"]


if "remitente" not in st.session_state:
    st.session_state.escenario = 0
    aplicar_escenario()


def leer_adjunto(archivo):
    if archivo is None:
        return None, None
    try:
        if archivo.name.lower().endswith(".pdf"):
            from pypdf import PdfReader
            texto = "\n".join((pagina.extract_text() or "") for pagina in PdfReader(archivo).pages)
        else:
            datos = archivo.getvalue()
            try:
                texto = datos.decode("utf-8")
            except UnicodeDecodeError:
                texto = datos.decode("cp1252", errors="replace")
    except Exception as error:
        return archivo.name, f"(no se pudo leer el adjunto: {error})"
    return archivo.name, texto.strip() or "(el adjunto no contiene texto extraíble; puede ser un PDF escaneado)"


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


def limpiar_markdown(texto):
    """Quita asteriscos de énfasis para mostrar y descargar la respuesta en texto plano."""
    return re.sub(r"\*{1,2}([^*\n]+?)\*{1,2}", r"\1", texto)


PATRON_RESUMEN = re.compile(r"^\s*(?:#+\s*|\*\*)?(?:1[.)]\s*)?(?:\*\*)?Resumen del correo", re.IGNORECASE)


def es_resumen(texto):
    """Reconoce el resumen de siete secciones aunque el modelo omita la numeración o use Markdown."""
    return bool(PATRON_RESUMEN.match(texto)) or ("Borrador de respuesta al cliente" in texto and "Prioridad sugerida" in texto)


def resumen_llamada(nombre, argumentos):
    a = argumentos
    if nombre == "crear_ticket_en_jira":
        return f"{a.get('titulo', '')} ({a.get('tipo', '')}, prioridad {a.get('prioridad', '')})"
    if nombre == "agendar_reunion_en_google_calendar":
        return f"{a.get('titulo', '')} · {a.get('fecha_inicio', '')}"
    if nombre == "consultar_disponibilidad_del_equipo":
        return f"del {a.get('fecha_inicio', '')} al {a.get('fecha_fin', '')}, {a.get('duracion_minutos', '')} min"
    if nombre == "actualizar_contacto_en_crm":
        return f"{a.get('nombre') or '(sin nombre)'} ({a.get('empresa', '')}) → {a.get('etapa', '')}"
    return ""


def resumen_salida(salida):
    o = salida["output"]
    if "error" in o:
        return f":red[error] {o['error']}"
    identificador = o.get("ticket") or o.get("evento_id") or o.get("contacto_id") or (plural(len(o["opciones"]), "opción libre", "opciones libres") if "opciones" in o else "")
    texto = " · ".join(t for t in (identificador, o.get("resultado") or o.get("estado") or "") if t)
    if o.get("aviso"):
        texto += f" · :orange[{o['aviso']}]"
    return texto


def mostrar_respuesta(texto):
    """Los resúmenes de siete secciones van en texto plano; las respuestas de chat, en Markdown."""
    if es_resumen(texto):
        st.text(limpiar_markdown(texto))
    else:
        st.markdown(texto)


def mostrar_evento(evento):
    """Dibuja un paso del ciclo del Run."""
    estado = evento["estado"]
    insignia = INSIGNIAS.get(estado, estado)
    if estado == "requires_action":
        st.markdown(f"{insignia} ciclo {evento['ciclo']}: el modelo pide {plural(len(evento['tool_calls']), 'función', 'funciones')} · `{evento['modelo']}`")
        for llamada in evento["tool_calls"]:
            nombre = llamada["function"]["name"]
            try:
                argumentos = json.loads(llamada["function"]["arguments"])
            except json.JSONDecodeError:
                argumentos = {"argumentos_invalidos": llamada["function"]["arguments"]}
            st.markdown(f"↳ `{nombre}`: {resumen_llamada(nombre, argumentos if isinstance(argumentos, dict) else {})}")
            st.json(argumentos, expanded=False)
    elif estado == "submit_tool_outputs":
        st.markdown(f"{insignia}: resultados devueltos al modelo.")
        for salida in evento["tool_outputs"]:
            st.markdown(f"↳ `{salida['funcion']}` → {resumen_salida(salida)}")
    else:
        st.markdown(f"{insignia}: {evento.get('detalle', '')}")


def mostrar_run(run):
    """Vuelve a dibujar un Run ya terminado (ciclo completo)."""
    final = run["registro"][-1]["estado"]
    with st.status(f"Run finalizado: {final}", state="error" if final == "failed" else "complete", expanded=True):
        for evento in run["registro"]:
            mostrar_evento(evento)


def correr(contenido, titulo, avisos=None):
    """Ejecuta un Run mostrando el avance en vivo y lo guarda en la sesión y en el hilo."""
    try:
        with st.status("Run en ejecución...", expanded=True) as estado:
            cliente = crear_cliente(api_key)
            texto, registro = ejecutar_run(cliente, st.session_state.hilo, contenido, al_avanzar=mostrar_evento,
                                           hilo_id=st.session_state.hilo_id)
            final = registro[-1]["estado"]
            estado.update(label=f"Run finalizado: {final}", state="error" if final == "failed" else "complete", expanded=True)
        st.session_state.runs.setdefault(st.session_state.hilo_id, []).append(
            {"titulo": titulo, "registro": registro, "respuesta": texto, "avisos": avisos or {}}
        )
    finally:  # el hilo se persiste aunque Streamlit interrumpa el Run
        guardar_hilo(st.session_state.hilo_id, st.session_state.get("cliente_hilo") or st.session_state.hilo_id, st.session_state.hilo)
    return texto


def mostrar_avisos(avisos):
    if avisos.get("enmascarados"):
        st.warning(f"Datos sensibles enmascarados antes de enviar el correo al modelo: {avisos['enmascarados']} (tarjetas, DNI o credenciales).")
    if avisos.get("senales"):
        st.error("El correo contiene frases que intentan dar instrucciones al asistente. Se marcó como incidente: " + "; ".join(avisos["senales"]))


def anotar_en_hilo(hilo_id, texto):
    """Agrega un mensaje del equipo al hilo indicado (el activo o uno guardado) y lo persiste."""
    if hilo_id == st.session_state.hilo_id:
        st.session_state.hilo.append({"role": "user", "content": PREFIJO_EQUIPO + texto})
        guardar_hilo(hilo_id, st.session_state.get("cliente_hilo") or hilo_id, st.session_state.hilo)
    else:
        datos = cargar_hilos().get(hilo_id, {})
        mensajes = datos.get("mensajes", []) + [{"role": "user", "content": PREFIJO_EQUIPO + texto}]
        guardar_hilo(hilo_id, datos.get("cliente") or hilo_id, mensajes)


# ---------- Barra lateral ----------
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("Clave de API (opcional si está en .env)", type="password",
                            help="Clave gratuita de Gemini: https://aistudio.google.com/apikey")
    st.caption(f"Proveedor: {NOMBRE_PROVEEDOR} · Modelo: `{MODELO}`")
    st.caption(f"Equipo de la cuenta: {', '.join(EQUIPO_CUENTA)}")
    st.caption("Acciones que requieren aprobación humana: " + "; ".join(ACCIONES_CON_APROBACION) + ".")

    st.divider()
    st.header("🧵 Hilos por cliente")
    hilos = cargar_hilos()
    opciones_hilo = ["(nuevo hilo)"] + list(hilos)
    st.session_state.selector_hilo = st.session_state.hilo_id if st.session_state.hilo_id in hilos else "(nuevo hilo)"
    st.selectbox("Hilo activo (Thread)", opciones_hilo, key="selector_hilo",
                 format_func=lambda h: h if h == "(nuevo hilo)" else f"{h} · {hilos[h]['cliente']}",
                 on_change=lambda: activar_hilo(None if st.session_state.selector_hilo == "(nuevo hilo)" else st.session_state.selector_hilo))
    st.caption(f"Mensajes en el hilo: {len(st.session_state.hilo)} (se envían al modelo los últimos {MAXIMO_MENSAJES}).")

    st.divider()
    st.checkbox("Borrar también la auditoría", key="borrar_auditoria")
    if st.button("🧹 Preparar demostración (vaciar sistemas e hilos)"):
        preparar_demostracion(borrar_auditoria=st.session_state.borrar_auditoria)
        activar_hilo(None)
        st.session_state.runs = {}
        st.rerun()

# ---------- Zona principal ----------
st.title("📨 UTP Assistant")
st.caption("Asistente de gestión de proyectos y ventas de UTPConsult · Tarea Académica 2 · Python + Streamlit + API compatible con OpenAI (Gemini)")

tab_correo, tab_chat, tab_sistemas, tab_registro = st.tabs(
    ["📥 Procesar correo", "💬 Conversar con el asistente", "🗂️ Sistemas simulados y aprobaciones", "🧾 Registro y auditoría"]
)

with tab_correo:
    col_izq, col_der = st.columns([1, 1.25])
    with col_izq:
        st.subheader("Correo entrante")
        st.selectbox("Escenario de ejemplo", range(len(ESCENARIOS)), key="escenario",
                     format_func=lambda i: ESCENARIOS[i]["nombre"], on_change=aplicar_escenario)
        st.caption("Resultado esperado: " + ESCENARIOS[st.session_state.escenario]["espera"])
        remitente = st.text_input("De", key="remitente")
        asunto = st.text_input("Asunto", key="asunto")
        cuerpo = st.text_area("Cuerpo del correo", key="cuerpo", height=200)
        archivo = st.file_uploader("Adjunto (txt, md o pdf)", type=["txt", "md", "pdf"],
                                   help="Arrastra un archivo o presiona el botón de carga. Si no adjuntas nada, marca la casilla del adjunto de ejemplo.")
        usar_ejemplo = st.checkbox("Usar el adjunto de ejemplo (requisitos iniciales de TechCorp)", key="usar_ejemplo")
        procesar = st.button("▶️ Procesar correo (crear Run)", type="primary")
    with col_der:
        st.subheader("Ciclo del Run")
        with st.popover("¿Qué significa cada estado?"):
            for estado, explicacion in LEYENDA:
                st.markdown(f"{INSIGNIAS[estado]} {explicacion}")
        respuesta = None
        if procesar:
            try:
                crear_cliente(api_key)  # se comprueba la clave antes de tocar el hilo
            except Exception as error:
                st.error(str(error))
                procesar = False
        if procesar:
            nombre_adjunto, texto_adjunto = leer_adjunto(archivo)
            if texto_adjunto is None and usar_ejemplo:
                nombre_adjunto, texto_adjunto = ADJUNTO_EJEMPLO.name, ADJUNTO_EJEMPLO.read_text(encoding="utf-8")
            contenido, avisos = formatear_correo(remitente, asunto, cuerpo, nombre_adjunto, texto_adjunto)
            hilo_id = obtener_hilo_id(remitente)
            if hilo_id != st.session_state.hilo_id:
                activar_hilo(hilo_id)  # ubica el Thread del cliente o crea uno nuevo
            if not st.session_state.hilo:
                st.session_state.cliente_hilo = remitente
            if any(m["role"] == "user" and avisos["message_id"] in (m.get("content") or "") for m in st.session_state.hilo):
                st.warning("Este correo ya fue procesado en este hilo (mismo Message-ID); no se crea otro Run.")
                procesar = False
        if procesar:
            mostrar_avisos(avisos)
            respuesta = correr(contenido, f"Correo: {asunto}", avisos)
            st.rerun()
        elif st.session_state.runs.get(st.session_state.hilo_id):
            ultimo = st.session_state.runs[st.session_state.hilo_id][-1]
            st.caption(f"Último Run del hilo {st.session_state.hilo_id} · {ultimo['titulo']}")
            mostrar_avisos(ultimo["avisos"])
            mostrar_run(ultimo)
            respuesta = ultimo["respuesta"]
            if respuesta is None:
                st.error("El Run falló. Revisa el último estado del registro; el hilo quedó como estaba antes de este Run.")
        else:
            st.info("Elige un escenario o pega un correo y presiona **Procesar correo**. Aquí verás cada estado del Run y, debajo, el resumen para el equipo.")
    if respuesta:
        st.subheader("Respuesta para el equipo interno")
        with st.container(border=True):
            mostrar_respuesta(respuesta)
        st.download_button("⬇️ Descargar la respuesta (.txt)", limpiar_markdown(respuesta), file_name=f"respuesta_{st.session_state.hilo_id}.txt")

with tab_chat:
    st.subheader("Conversación con el asistente dentro del hilo")
    historial = st.container()
    with historial:
        for mensaje in st.session_state.hilo:
            contenido = mensaje.get("content") or ""
            if mensaje["role"] == "user":
                with st.chat_message("user"):
                    if contenido.startswith("CORREO ENTRANTE"):
                        with st.expander("📧 Correo procesado (clic para ver)"):
                            st.text(contenido)
                    else:
                        st.markdown(contenido.removeprefix(PREFIJO_EQUIPO))
            elif mensaje["role"] == "assistant" and contenido:
                with st.chat_message("assistant"):
                    if es_resumen(contenido):
                        with st.expander("📋 Resumen del Run para el equipo (clic para ver)"):
                            st.text(limpiar_markdown(contenido))
                    else:
                        st.markdown(contenido)
    ultimo_chat = (st.session_state.runs.get(st.session_state.hilo_id) or [None])[-1]
    if ultimo_chat and ultimo_chat["respuesta"] is None:
        st.error(f"El último Run de este hilo falló: {ultimo_chat['registro'][-1].get('detalle', '')} Vuelve a enviar la instrucción.")
    pregunta = st.chat_input("Pregunta o instrucción del equipo (por ejemplo: ¿qué queda pendiente con TechCorp?)")
    if pregunta:
        try:
            crear_cliente(api_key)
        except Exception as error:
            st.error(str(error))
            pregunta = None
    if pregunta:
        if not st.session_state.hilo_id:
            activar_hilo("thread_equipo")
            st.session_state.cliente_hilo = "equipo interno"
        with historial:
            with st.chat_message("user"):
                st.markdown(pregunta)
            with st.chat_message("assistant"):
                titulo = f"Instrucción: {pregunta[:40].rstrip()}{'…' if len(pregunta) > 40 else ''}"
                respuesta_chat = correr(PREFIJO_EQUIPO + pregunta, titulo)
                if respuesta_chat:
                    mostrar_respuesta(respuesta_chat)
        st.rerun()

with tab_sistemas:
    st.info(
        "Jira, Google Calendar y el CRM están simulados: no hay cuentas reales. Cada vez que el modelo pide una "
        "función, asistente.py valida los argumentos, aplica las reglas de negocio y guarda el resultado en un "
        "archivo JSON de la carpeta datos/. Las funciones reciben los mismos argumentos que recibiría la integración real, "
        "así que reemplazar la simulación por las API reales no cambia el resto del código."
    )
    tickets, eventos, contactos = estado_sistemas()
    c1, c2, c3 = st.columns(3)
    c1.metric("Tickets en Jira", len(tickets))
    c2.metric("Eventos en el calendario", len(eventos))
    c3.metric("Contactos en el CRM", len(contactos))

    st.subheader("Bandeja de aprobaciones")
    tentativos = [e for e in eventos if e["estado"].startswith("tentativo")]
    if not tentativos:
        st.caption("No hay reuniones pendientes de aprobación.")
    for e in tentativos:
        with st.container(border=True):
            invitados = ", ".join(str(a) for a in e["asistentes_cliente"]) or "ninguno"
            st.markdown(f"**{e['titulo']}** · {formatear_fecha(e['inicio'])} · {e['modalidad']} · invitados del cliente: {invitados}")
            st.caption(f"Agenda: {e['agenda'] or 'sin agenda'} · evento {e['evento_id']} · hilo {e.get('hilo_id') or 'sin asignar'}")
            b1, b2, _ = st.columns([2, 1, 2])
            aprobado = b1.button("✅ Aprobar y enviar invitaciones", key=f"aprobar_{e['evento_id']}")
            rechazado = b2.button("❌ Rechazar", key=f"rechazar_{e['evento_id']}")
            if aprobado or rechazado:
                texto = aprobar_evento(e["evento_id"], aprobado)
                destino = e.get("hilo_id") or st.session_state.hilo_id  # el hilo del cliente dueño del evento
                if destino:
                    anotar_en_hilo(destino, texto)
                registrar_auditoria({"tipo": "aprobacion", "evento_id": e["evento_id"], "aprobado": aprobado, "hilo_id": destino})
                st.rerun()

    for titulo, filas, orden in (("Jira (simulado)", tickets, None), ("Google Calendar (simulado)", eventos, ORDEN_CALENDARIO), ("CRM (simulado)", contactos, None)):
        st.subheader(titulo)
        if filas:
            st.dataframe(aplanar(filas), width="stretch", column_order=orden)
        else:
            st.caption("Sin registros todavía.")

with tab_registro:
    st.subheader("Auditoría persistente (datos/auditoria.jsonl)")
    auditoria = leer_auditoria()
    if auditoria:
        st.dataframe([{
            "Fecha y hora": a["fecha_hora"], "Tipo": a["tipo"], "Run": a.get("run_id") or "", "Evento": a.get("evento_id") or "",
            "Aprobado": a.get("aprobado"), "Hilo": a.get("hilo_id") or "", "Estado final": a.get("estado_final") or "",
            "Duración (s)": f"{a['duracion_s']:.1f}" if a.get("duracion_s") is not None else "", "Funciones": ", ".join(f["nombre"] for f in a.get("funciones", [])),
            "Incidente": a.get("incidente"), "Modelo": a.get("modelo") or "", "Error": a.get("error") or "",
        } for a in reversed(auditoria)], width="stretch", column_config={"Tipo": st.column_config.TextColumn(width="small")})
    else:
        st.caption("Todavía no hay registros de auditoría.")
    st.subheader("Runs de esta sesión")
    runs_sesion = [(hilo_id, run) for hilo_id, lista in st.session_state.runs.items() for run in lista]
    if not runs_sesion:
        st.caption("Todavía no se ejecutó ningún Run en esta sesión.")
    for numero, (hilo_id, run) in enumerate(runs_sesion, start=1):
        with st.expander(f"Run {numero} · hilo {hilo_id} · {run['titulo']} · estado final: {run['registro'][-1]['estado']}"):
            for evento in run["registro"]:
                mostrar_evento(evento)
