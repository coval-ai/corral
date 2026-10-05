# benchmark-harness

Coval's own agent for the S2S leaderboard. A Pipecat pipeline runs a cascaded
STT, LLM, TTS stack (native S2S models later) behind a WebSocket that Coval's
simulator calls, measures each turn where the model is, and reports it to Coval
as a span. One command runs a benchmark from a laptop with a Coval API key, and
the same image runs it as a job.

Design: `internal-docs/design-docs/benchmarks-s2s-pipecat-harness.md`
([coval-ai/internal-docs#235](https://github.com/coval-ai/internal-docs/pull/235)).
Linear project: S2S Pipecat harness.

## Setup

```bash
brew install portaudio cloudflared     # mic mode; local tunnel (no account)
uv sync --extra local
cp .env.example .env                   # provider and Coval keys
```

Keys come from the repo root's `.env`. Pipecat is pinned `~=1.12`.

## Stacks

A file under `src/agent/stacks/` names one provider and model per role plus
turn-taking and sample rates. Vendors are a table in `src/agent/services/`:
a Pipecat class per role and the key it uses; adding one is a line there and
its Pipecat extra in `pyproject.toml`. The scenario prompt comes from
`src/agent/scenarios/<scenario>/`, with a `system-prompt.<stack>.txt` override
when a model needs its own; stack and prompt are hashed together into the
digest every run and span carries.

## Running

`uv run agent` with no arguments asks for a mode, then a stack:

```
1. talk       microphone and speakers on this machine
2. benchmark  serve behind a tunnel, run it on Coval, exit
Mode [1-2]:
```

With a mode it is a script. Every mode takes `--stack SLUG` and
`--scenario NAME`, defaulting to `S2S_STACK` and `S2S_SCENARIO`.

`agent talk` talks to a stack on this machine's microphone and speakers;
it needs the `local` extra. The agent speaks first; Ctrl-C hangs up. Per-turn
latency is logged.

`agent benchmark [--test-set NAME] [--smoke]` runs one benchmark from
wherever it is started, a laptop or a cloud job: it serves in-process behind a
Cloudflare quick tunnel, probes it over HTTP and one authenticated call,
upserts the stack's Coval WebSocket agent at that URL, launches a run on the
named test set with its persona and metrics at concurrency 1, polls it to a
terminal status, prints each metric's values, and exits. Nothing is left
running. It needs `COVAL_API_KEY` and `cloudflared` on the path; `--no-tunnel`
probes localhost. A quick tunnel's hostname takes about 80 s to resolve.
`--smoke` runs a single test case to check the wiring end to end.

Test sets are named in `src/agent/scenarios/<scenario>/test-sets.json`,
each with its Coval test set, persona and metric ids; `baseline` is the
default. Interactively, the set is the second question, after the mode.

`agent serve` is the image's command: the same pipeline behind a WebSocket
on `PORT` (default 8080) with no tunnel and no run, one call per connection,
speaking the wire contract in `coval.py`: raw PCM both ways, `session_ready`
once every service is up, bearer from `S2S_AGENT_TOKEN`.

## Measurement

Every call sends one `s2s.turn` span per answered turn to Coval's trace ingest
under its simulation id: `s2s.provider_latency_ms` (caller stopped to first
synthesized audio) and `s2s.harness_overhead_ms` (to the first byte written
back), for a custom trace metric to aggregate. T0 is Coval's `utterance_end`
marker when present, else the local VAD stop.

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src tests
uv run pytest -q --disable-socket --allow-unix-socket --allow-hosts=127.0.0.1,::1
```
