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

`uv run agent` with no arguments asks mode, scenario, stack:

```
1. talk       to the agent
2. benchmark  with Coval
Mode [1-2, default 1]:
```

Enter takes the first option at every prompt.

Talk lists every scenario; benchmark lists the ones with Coval ids. With a
mode it is a script: every mode takes `--stack SLUG` and `--scenario NAME`,
defaulting to `S2S_STACK` and `S2S_SCENARIO`, and without those to `default`
for talk and `baseline` otherwise.

`agent talk` talks to a stack on this machine's microphone and speakers; it
needs the `local` extra. The agent speaks first; Ctrl-C hangs up. Per-turn
latency is logged.

`agent benchmark [--smoke]` runs one benchmark from wherever it is started, a
laptop or a cloud job: it serves in-process behind a Cloudflare quick tunnel,
probes it over HTTP and one authenticated call, upserts the scenario's Coval
WebSocket agent at that URL, launches a run on the scenario's test set with its
persona and metrics at concurrency 1, polls it to a terminal status, prints
each metric's values, and exits. Nothing is left running. It needs
`COVAL_API_KEY` and `cloudflared` on the path; `--no-tunnel` probes localhost.
A quick tunnel's hostname takes about 80 s to resolve. `--smoke` runs a single
test case to check the wiring end to end.

`agent serve` is the image's command: the same pipeline behind a WebSocket on
`PORT` (default 8080) with no tunnel and no run, one call per connection,
speaking the wire contract in `coval.py`: raw PCM both ways, `session_ready`
once every service is up, bearer from `S2S_AGENT_TOKEN`.

## Measurement

Every answered turn sends one `turn` span to Coval under the call's
simulation id, with `stt`, `llm` and `tts` child spans for the three stages,
each tagged with the provider and model that ran it. We write down the clock
at six moments and put them all on the turn as `s2s.<moment>.unix_ms`:

- `utterance_end`: Coval tells us the caller finished talking
- `vad_done`: our VAD and smart-turn agree the caller is done
- `stt_done`: the last transcript the model actually got to read
- `llm_first_token`: the model says its first word
- `tts_first_audio`: we have the first bit of synthesized audio
- `audio_start`: we send the first audio chunk back to Coval

The time between them goes on as `s2s.<step>_ms`: `latency` is the headline
(caller stopped to first synthesized audio), then `endpointing`, `stt`, `llm`,
`tts` and `transport` break it down. "Caller stopped" means `utterance_end` if
Coval sent it, otherwise `vad_done`; `s2s.latency.source` tells you which.

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src tests
uv run pytest -q --disable-socket --allow-unix-socket --allow-hosts=127.0.0.1,::1
```
