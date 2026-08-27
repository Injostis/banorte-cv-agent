# Agente de CV — Rodrigo Rios

Agente conversacional que responde preguntas sobre mi trayectoria profesional
(experiencia, proyectos, habilidades, educación), construido para el Reto IA
Banorte. Expone un endpoint HTTP compatible con el protocolo
[Open Responses](https://www.openresponses.org/), el contrato que usa la
plataforma del reto para hablar con cualquier agente que registres.

## Qué hace

Contesta con base en un perfil estructurado (`data/profile.yaml`), nunca
inventando datos — si algo no está ahí, lo dice. Tiene un guardrail de
entrada que solo deja pasar preguntas sobre mi trayectoria (más un dato
curioso personal, ver más abajo) y rechaza intentos de manipular sus
instrucciones.

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
                                  |     └─ tools leen profile.yaml / ps_trophies.json
                                  └─ arma la respuesta en formato Open Responses
```

Todo corre en un solo proceso, sin más servicios externos que la API de
Anthropic. `app/responses_schema.py` es la única pieza que sabe que existe
el protocolo Open Responses — el resto trabaja con texto plano.

## Decisiones técnicas

- **Un solo agente, sin orquestador.** El perfil es un puñado de datos
  conocidos de antemano, no algo que justifique un sistema multiagente.
- **Sin RAG.** No hay un corpus grande donde buscar por similitud tenga
  sentido — las tools regresan exactamente el bloque de información que se
  les pide, siempre.
- **Sin estado, sin base de datos.** Cada request se resuelve solo con lo
  que trae, sin depender de nada guardado de antes.
- **Guardrail en dos capas.** Un filtro de regex rápido primero, y si no
  encuentra nada, una llamada a Claude que clasifica intento de
  manipulación y si el tema es válido. Si esa llamada falla, se rechaza —
  nunca se asume que pasó.

## Cómo se verificó

24 tests (`tests/`) cubren el contrato HTTP, el guardrail y las tools, con
las llamadas a Claude mockeadas para que corran rápido.

`scripts/eval_agent.py` corre una batería de 15 preguntas reales contra el
servicio de verdad, con la API real — puntuales, amplias, fuera de tema, un
intento de injection, seguimiento ambiguo. No es solo cosmético: la primera
corrida encontró un problema real que los tests unitarios no atrapan (un
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
framework de agentes encima), Pydantic, `uv` para dependencias, Docker.

## Estructura

```
app/
  main.py              endpoint HTTP, auth, orquesta guardrail + agente
  agent.py             system prompt y loop de tool-use
  guardrails.py        guardrail de entrada (regex + clasificador)
  tools.py             las tools que el agente puede llamar
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
tests/                 24 tests, API mockeada
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
