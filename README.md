# 📨 UTP Assistant

Asistente de IA para el equipo de gestión de proyectos y ventas de la consultora UTPConsult. Lee los correos de los clientes, registra lo importante en Jira, Google Calendar y el CRM, y deja al equipo un resumen listo para actuar. Ninguna acción con impacto externo se ejecuta sin que una persona la apruebe.

Tarea Académica 2 del curso Herramientas de Desarrollo Profesional - TIC (UTP). Integrantes: Leonardo Martinez Concha y Kevin Julio Chacaltana Vargas.

## Qué hace

1. Recibe un correo entrante (remitente, asunto, cuerpo y un adjunto opcional en txt, md o pdf). Antes de enviarlo al modelo enmascara tarjetas, DNI y credenciales, y detecta frases que intentan dar instrucciones al asistente.
2. Ubica el hilo (Thread) de ese cliente, identificado por el dominio del remitente y guardado en `datos/hilos.json`, para que el asistente tenga el contexto de los correos anteriores.
3. Ejecuta el ciclo de un **Run**: el modelo analiza el correo y el hilo, pide las funciones que necesita (`requires_action`), el sistema valida los argumentos contra el esquema, aplica las reglas de negocio, ejecuta las funciones y devuelve los resultados (`submit_tool_outputs`), y el modelo redacta la respuesta final (`completed`).
4. Entrega al equipo un resumen de siete secciones: resumen del correo, contacto, requisitos detectados, acciones ejecutadas, pendientes, borrador de respuesta al cliente y prioridad sugerida.
5. Permite conversar con el asistente dentro del hilo, aprobar o rechazar las reuniones tentativas desde la bandeja de aprobaciones y revisar los sistemas simulados y la auditoría.

## Herramientas que puede invocar (function calling)

| Función | Para qué sirve |
|---------|----------------|
| `crear_ticket_en_jira` | Registra un requisito, tarea o épica con su cita de origen; no duplica títulos del mismo cliente |
| `consultar_disponibilidad_del_equipo` | Busca espacios libres del equipo (días hábiles, horario laboral, sin cruces con eventos existentes) |
| `agendar_reunion_en_google_calendar` | Crea el evento, siempre tentativo y sin invitaciones hasta que una persona lo apruebe |
| `actualizar_contacto_en_crm` | Crea o actualiza el contacto, su etapa comercial y una nota; la etapa "perdido" no se aplica de forma automática (se conserva la anterior y una persona la cambia en el CRM) |

Cada correo recibe un Message-ID calculado a partir del remitente, el asunto y el cuerpo; si el mismo correo se procesa dos veces en un hilo, la aplicación lo avisa y no crea otro Run.

## ¿Qué significa "sistemas simulados"?

UTPConsult y sus cuentas de Jira, Google Calendar y CRM son ficticias, así que las cuatro funciones no llaman a las API reales: guardan sus datos en archivos JSON de la carpeta `datos/` (`jira.json`, `calendario.json`, `crm.json`). Reciben exactamente los mismos argumentos que recibiría la integración real y devuelven resultados con el mismo formato (clave del ticket, id del evento, id del contacto), por lo que el modelo no nota la diferencia. Para conectar el sistema real solo hay que reemplazar el cuerpo de cada función; el prompt, los esquemas y el ciclo del Run no cambian.

La carpeta `datos/` también guarda los hilos por cliente (`hilos.json`) y la auditoría (`auditoria.jsonl`, una línea por Run y por aprobación). Se crea sola al usar la aplicación y no se sube al repositorio.

## Diseño y tecnología

El diseño sigue la arquitectura de la API de Asistentes de OpenAI (Assistant, Thread, Run, `requires_action`, `submit_tool_outputs`). Como esa API es de pago, la implementación usa la **API gratuita de Google Gemini** a través de su endpoint compatible con OpenAI, con la misma librería `openai` de Python:

| Concepto del diseño | En el código |
|---------------------|--------------|
| Assistant (instrucciones, modelo, herramientas) | `PROMPT_SISTEMA`, `MODELO` y `HERRAMIENTAS` en `asistente.py` |
| Thread | Un hilo por cliente en `datos/hilos.json`, referenciado desde el contacto del CRM |
| Run | `ejecutar_run()`; si falla, el hilo vuelve a su estado previo y queda registrado como `failed` |
| `additional_instructions` | `contexto_de_ejecucion()`: fecha y hora de Lima, equipo de la cuenta e hilo activo |
| `requires_action` / `tool_calls` | `message.tool_calls` que devuelve el modelo |
| Validación y `submit_tool_outputs` | `validar_argumentos()` + `ejecutar_funcion()`; los resultados vuelven como mensajes `role: tool` |
| `file_search` sobre el adjunto | El texto del adjunto se incorpora al mensaje del correo, delimitado entre `<<<` y `>>>` |
| Run Steps y auditoría | Registro de estados en vivo en la interfaz y `datos/auditoria.jsonl` |
| Bandeja de aprobación | Pestaña "Sistemas simulados y aprobaciones": `aprobar_evento()` confirma o rechaza y el hilo recibe el mensaje del equipo |

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

La clave de Gemini se obtiene gratis en <https://aistudio.google.com/apikey>. La capa gratuita limita las peticiones por día y por modelo; si el modelo principal agota su cuota, el asistente pasa solo a los modelos de respaldo definidos en el `.env` y lo indica en el ciclo del Run. También se puede pegar la clave en la barra lateral de la aplicación.

## Estructura

```
UTP-ASSISTANT/
├── app.py                     # Interfaz en Streamlit (correo, ciclo del Run, chat, aprobaciones, auditoría)
├── asistente.py               # Prompt de sistema, esquemas, validación, sistemas simulados, hilos, auditoría y ciclo del Run
├── ejemplos/escenarios.json   # Seis correos de ejemplo para la demostración
├── ejemplos/requisitos_iniciales.txt  # Adjunto de ejemplo (requisitos del módulo de pagos de TechCorp)
├── requirements.txt
├── .env.example
└── .streamlit/config.toml
```

## Prueba rápida

Presiona "Preparar demostración" en la barra lateral y recorre los escenarios del selector, en este orden:

1. Aceptación con fecha ambigua y adjunto (TechCorp): contacto en el CRM, tickets por requisito y tres fechas propuestas sin agendar en firme.
2. Confirmación de fecha en el mismo hilo (TechCorp): evento tentativo; apruébalo en la bandeja y vuelve al chat para ver el mensaje del equipo.
3. Requisitos vagos sin adjunto (Acme Perú): una sola épica con preguntas y ninguna reunión.
4. Queja con plazo corto (Clínica Monte Azul): ticket de tipo Error con prioridad alta y reunión tentativa.
5. Intento de manipulación con datos sensibles: datos enmascarados, incidente reportado y ninguna función ejecutada.
6. Correo sin acción (boletín): ninguna función y prioridad baja.

Luego escríbele al asistente en la pestaña de conversación, por ejemplo: "¿qué queda pendiente con TechCorp?".
