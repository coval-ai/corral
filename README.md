# benchmark-harness

Coval's own voice agent for the S2S leaderboard. It runs a speech-to-text,
LLM, text-to-speech stack in Pipecat, lets Coval's simulator call it, and
times every turn from inside the pipeline so we know which model the pause
came from.

## Setup

```bash
uv sync
cp .env.example .env    # then fill in your keys
```

## Run

```bash
uv run agent
```

Opens a tui that helps set up the agent. Defaults to a local test.

```
1. talk       to the agent
2. benchmark  with Coval
Mode [1-2, default 1]:
```

**talk** puts the models on your mic and speakers. The agent greets you, you
talk, Ctrl-C hangs up. Latency per turn is logged. Needs a mic, so first:
`brew install portaudio && uv sync --extra local`.

**benchmark** runs one Coval benchmark and exits. It serves the agent behind a
Cloudflare tunnel, points Coval's agent record at it, launches a run on the
scenario's test set, waits for it to finish, and prints the metrics. Nothing
stays running. Expect about 80 seconds of waiting for the tunnel hostname
before anything happens. `--smoke` runs a single test case. Needs
`brew install cloudflared`.

**serve** is what the Docker image runs: the agent on a port, no tunnel, no
run.

For scripts, pass the mode and flags instead of answering prompts:

```bash
uv run agent benchmark --scenario baseline --stt soniox/stt-rt-v5 --llm google/gemini-2.5-flash --smoke
```

Leave a modality out and it gets the first model on the list. `S2S_STT`, `S2S_LLM`,
`S2S_TTS` and `S2S_SCENARIO` set the defaults. Benchmarks need
`COVAL_API_KEY`; talk needs the provider keys for the models you picked. If
something is missing, the menu says so and gives the command.

## What gets measured

Every answered turn becomes a `turn` span in Coval, with `stt`, `llm` and
`tts` spans under it showing how long each stage took and which provider and
model ran it. We grab the time at seven moments:

- `utterance_end`: Coval tells us the caller finished talking
- `vad_stop`: Silero hears enough silence and Pipecat asks the STT to finalize
- `stt_done`: the STT sends back the transcript the model will read
- `vad_done`: smart-turn agrees and the transcript is in, so the turn is over
- `llm_first_token`: the model says its first word
- `tts_first_audio`: we have the first bit of synthesized audio
- `audio_start`: we send the first audio chunk back to Coval

Each one is on the span as `s2s.<moment>.unix_ms`, and the gaps between them
as `s2s.<step>_ms`. The headline is `s2s.latency_ms`: caller stopped to first
synthesized audio. "Caller stopped" is Coval's `utterance_end` marker on Coval
runs. Locally it's when Silero fired minus its silence window, which is when
you actually stopped. `s2s.latency.source` says which.

`s2s.stt_ms` is `vad_stop` to `stt_done`, how long the STT took to answer the
finalize. Every STT in the catalog runs with its own endpointing off so they're
all measured the same way. On Coval runs, `s2s.endpointing_ms` is `utterance_end` to
`vad_stop`, how long after the caller stopped Silero noticed.

Known caveat: Coval currently sends `utterance_end` about 350 ms after the
persona actually stops, so the headline reads about 350 ms low on every stack.
Comparisons between stacks are fine; absolute numbers are not yet.

## Adding things

A **model** is an entry in `src/agent/models/stt.json`, `llm.json` or `tts.json`.
The first in each file is the default. If the provider is new, add its Pipecat class to the
table in `src/agent/models/__init__.py` and its extra to `pyproject.toml`.

A **scenario** is a directory in `src/agent/scenarios/`. See the README there.

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src tests
uv run pytest -q --disable-socket --allow-unix-socket --allow-hosts=127.0.0.1,::1
```
