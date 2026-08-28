# Agente de CV — Rodrigo Rios

Agente conversacional que responde preguntas sobre mi trayectoria profesional
(experiencia, proyectos, habilidades, educación), construido para el Reto IA
Banorte. Expone un endpoint HTTP compatible con el protocolo
[Open Responses](https://www.openresponses.org/), el contrato que usa la
plataforma del reto para hablar con cualquier agente que registres.

## Qué hace

Contesta con base en un perfil estructurado (`data/profile.yaml`), nunca
inventando datos — si algo no está ahí, lo dice. Además de texto:

- Muestra tarjetas visuales ([A2UI](https://a2ui.org)) para el perfil, las
  skills por nivel y los trofeos de PlayStation, con respaldo en texto
  siempre por si el cliente no puede renderizarlas.
- Acepta imágenes en la entrada (las comenta, nunca sigue instrucciones que
  vengan de dentro de una).
- Responde en modo normal o streaming (SSE), según lo pida el cliente.

Tiene dos guardrails: uno de entrada (solo deja pasar preguntas sobre mi
trayectoria y rechaza intentos de manipular sus instrucciones) y uno de
salida (verifica que la respuesta esté respaldada por los datos que la
tool trajo ese turno, antes de mandarla).

No es un sistema RAG, ni tiene un orquestador tipo LangGraph, ni base de
datos — decisiones a propósito, no por default.

## Cómo está armado

```
Banorte  --POST /responses-->  FastAPI
                                  |
                                  ├─ verifica el token Bearer
                                  ├─ guardrail de entrada (regex + Claude)
                                  |     └─ si rechaza, responde ahí mismo
                                  ├─ loop de tool-use con Claude
                                  |     ├─ tools leen profile.yaml / ps_trophies.json
                                  |     └─ algunas regresan una superficie A2UI
                                  ├─ guardrail de salida (verifica lo que se
                                  |     va a decir contra los datos usados)
                                  └─ arma la respuesta (Open Responses,
                                        normal o streaming)
```

Todo corre en un solo proceso, sin más servicios externos que la API de
Anthropic (y Langfuse, opcional, para trazas). `app/responses_schema.py` es
la única pieza que sabe que existe el protocolo Open Responses — el resto
trabaja con texto plano y estructuras propias.

## Decisiones técnicas

- **Un solo agente, sin orquestador.** El perfil es un puñado de datos
  conocidos de antemano, no algo que justifique un sistema multiagente.
- **Sin RAG.** No hay un corpus grande donde buscar por similitud tenga
  sentido — las tools regresan exactamente el bloque de información que se
  les pide, siempre.
- **Sin estado, sin base de datos.** Cada request se resuelve solo con lo
  que trae, sin depender de nada guardado de antes.
- **Guardrail de entrada en dos capas.** Un filtro de regex rápido primero,
  y si no encuentra nada, una llamada a Claude que clasifica intento de
  manipulación y si el tema es válido. Si esa llamada falla, se rechaza —
  nunca se asume que pasó (fail-closed).
- **Guardrail de salida, pero fail-open.** Verifica que la respuesta final
  no invente nada fuera de los datos que las tools trajeron ese turno. A
  diferencia del de entrada, un fallo al clasificar aquí no bloquea la
  respuesta: es una verificación de calidad, no de seguridad, y tumbar una
  respuesta válida por un error transitorio de red cuesta más de lo que
  el riesgo justifica.
- **A2UI solo con el catálogo básico.** Nada de componentes exóticos sin
  verificar — cada superficie se probó en vivo contra la plataforma real
  antes de darla por buena.

## Cómo se verificó

53 tests (`tests/`) cubren el contrato HTTP, ambos guardrails, las tools y
las superficies A2UI, con las llamadas a Claude mockeadas para que corran
rápido y sin costo.

`scripts/eval_agent.py` corre una batería de preguntas reales contra el
servicio de verdad, con la API real — puntuales, amplias, fuera de tema, un
intento de injection, seguimiento ambiguo. No es solo cosmético: encontró
más de un problema real que los tests unitarios no atrapan (por ejemplo, un
simple "hola" se estaba rechazando como fuera de tema), y quedó corregido
antes de seguir.

## El dato curioso: mis platinos de PlayStation

`data/ps_trophies.json` no es una lista que escribí a mano — es un snapshot
generado una sola vez por `scripts/fetch_ps_trophies.py`, que se conecta a
la API de PlayStation Network (no hay una oficial; usé
[PSNAWP](https://github.com/isFakeAccount/psnawp)) y trae, por cada
platino, el juego, la plataforma, la fecha y el porcentaje de jugadores en
el mundo que también lo tiene.

La corrí una sola vez, a mano — el servicio desplegado nunca llama a PSN.
Un uso repetido o automatizado puede terminar en una suspensión de cuenta
(lo advierte la propia librería), así que separé las dos cosas: la ingesta
corre una vez y deja un archivo; el agente en producción solo lo lee.

## Stack

Python 3.13, FastAPI, SDK oficial de `anthropic` (tool use nativo, sin
framework de agentes encima), Pydantic, `uv` para dependencias, Docker,
Langfuse para trazas/costos (opcional).

## Estructura

```
app/
  main.py              endpoint HTTP, auth, agent card, orquesta todo
  agent.py             system prompt, loop de tool-use, guardrail de salida
  guardrails.py        guardrail de entrada y de salida
  tools.py             las tools que el agente puede llamar
  a2ui.py               construye las superficies A2UI (perfil, skills, trofeos)
  images.py            soporte de imágenes en la entrada
  observability.py     integración con Langfuse
  responses_schema.py  traducción hacia/desde el formato Open Responses
  profile_data.py      carga profile.yaml
  ps_trophies_data.py  carga ps_trophies.json
  config.py            variables de entorno
data/
  profile.yaml         fuente de verdad del perfil
  ps_trophies.json     snapshot de trofeos (ver arriba)
scripts/
  fetch_ps_trophies.py ingesta única contra la API de PSN
  eval_agent.py        batería de preguntas reales contra el servicio
tests/                 53 tests, API mockeada
```

## Correrlo en local

```bash
uv sync
cp .env.example .env   # y completa ANTHROPIC_API_KEY + AGENT_BEARER_TOKEN
uv run uvicorn app.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/responses \
  -H "Authorization: Bearer <tu-token>" \
  -H "Content-Type: application/json" \
  -d '{"input": "¿en qué trabajas actualmente?"}'
```

Verificación completa antes de un cambio:

```bash
uv run pytest
uv run ruff check .
uv run mypy app scripts --strict
uv run python scripts/eval_agent.py   # con el servicio corriendo
```
