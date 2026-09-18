# 📨 UTP Assistant

Asistente de IA para el equipo de gestión de proyectos y ventas de la consultora UTPConsult. Lee los correos de los clientes, registra lo importante en Jira, Google Calendar y el CRM, y deja al equipo un resumen listo para actuar.

Tarea Académica 2 del curso Herramientas de Desarrollo Profesional - TIC (UTP). Integrantes: Leonardo Martinez Concha y Kevin Julio Chacaltana Vargas.

## Qué hace

1. Recibe un correo entrante (remitente, asunto, cuerpo y un adjunto opcional en txt, md o pdf).
2. Ejecuta el ciclo de un **Run**: el modelo analiza el correo y el hilo, pide las funciones que necesita (`requires_action`), el sistema las ejecuta y le devuelve los resultados (`submit_tool_outputs`), y el modelo redacta la respuesta final (`completed`).
3. Entrega al equipo un resumen de siete secciones: resumen del correo, contacto, requisitos detectados, acciones ejecutadas, pendientes, borrador de respuesta al cliente y prioridad sugerida.
4. Permite conversar con el asistente dentro del mismo hilo y revisar los sistemas simulados.

## Herramientas que puede invocar (function calling)

| Función | Para qué sirve |
|---------|----------------|
| `crear_ticket_en_jira` | Registra un requisito, tarea o épica con su cita de origen |
| `consultar_disponibilidad_del_equipo` | Busca espacios libres del equipo antes de proponer una reunión |
| `agendar_reunion_en_google_calendar` | Crea el evento (tentativo hasta que una persona lo apruebe) |
| `actualizar_contacto_en_crm` | Crea o actualiza el contacto, su etapa comercial y una nota |

Jira, Google Calendar y el CRM están simulados: cada función guarda sus datos en `datos/*.json`. Las firmas son las mismas que tendrían con las API reales.

## Diseño y tecnología

El diseño sigue la arquitectura de la API de Asistentes de OpenAI (Assistant, Thread, Run, `requires_action`, `submit_tool_outputs`). Como esa API es de pago, la implementación usa la **API gratuita de Google Gemini** a través de su endpoint compatible con OpenAI, con la misma librería `openai` de Python:

| Concepto del diseño | En el código |
|---------------------|--------------|
| Assistant (instrucciones, modelo, herramientas) | `PROMPT_SISTEMA`, `MODELO` y `HERRAMIENTAS` en `asistente.py` |
| Thread | `st.session_state.hilo` (correos, llamadas y resultados) |
| Run | `ejecutar_run()` |
| `requires_action` / `tool_calls` | `message.tool_calls` que devuelve el modelo |
| `submit_tool_outputs` | Mensajes con `role: tool` y `tool_call_id` |
| `file_search` sobre el adjunto | El texto del adjunto se incorpora al mensaje del correo |
| Run Steps | Registro de estados que la interfaz muestra en vivo |

## Instalación y ejecución

```bash
git clone https://github.com/Linkingfor/UTP-ASSISTANT.git
cd UTP-ASSISTANT
python -m venv .venv
.venv\Scripts\activate        # En Linux o Mac: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env        # Luego pega tu GEMINI_API_KEY dentro de .env
streamlit run app.py
```

La clave de Gemini se obtiene gratis en <https://aistudio.google.com/apikey>. La capa gratuita limita las peticiones por día y por modelo; si el modelo principal agota su cuota, el asistente pasa solo a los modelos de respaldo definidos en el `.env`. También se puede pegar la clave en la barra lateral de la aplicación.

## Estructura

```
UTP-ASSISTANT/
├── app.py                        # Interfaz en Streamlit (correo, ciclo del Run, chat, sistemas)
├── asistente.py                  # Prompt de sistema, esquemas de funciones, sistemas simulados y ciclo del Run
├── ejemplos/requisitos_iniciales.txt  # Adjunto de ejemplo (requisitos del módulo de pagos de TechCorp)
├── requirements.txt
├── .env.example
└── .streamlit/config.toml
```

## Prueba rápida

La aplicación carga por defecto el correo de ejemplo de Ana Torres (TechCorp) con su adjunto. Pulsa **Procesar correo (crear Run)** y observa cada estado del Run, las funciones que pide el modelo, los resultados y el resumen final. Luego puedes escribirle al asistente en la pestaña de conversación, por ejemplo: "¿qué queda pendiente con TechCorp?".
