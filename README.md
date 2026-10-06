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

**talk** puts a stack on your mic and speakers. The agent greets you, you
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
uv run agent benchmark --scenario baseline --stack cascade-soniox-gemini --smoke
```

`S2S_STACK` and `S2S_SCENARIO` set the defaults. Benchmarks need
`COVAL_API_KEY`; talk needs the provider keys for the stack you picked. If
something is missing, the menu says so and gives the command.

## What gets measured

Every answered turn becomes a `turn` span in Coval, with `stt`, `llm` and
`tts` spans under it showing how long each stage took and which provider and
model ran it. We grab the time at six moments:

- `utterance_end`: Coval tells us the caller finished talking
- `vad_done`: our VAD and smart-turn agree the caller is done
- `stt_done`: the last transcript the model actually got to read
- `llm_first_token`: the model says its first word
- `tts_first_audio`: we have the first bit of synthesized audio
- `audio_start`: we send the first audio chunk back to Coval

Each one is on the span as `s2s.<moment>.unix_ms`, and the gaps between them
as `s2s.<step>_ms`. The headline is `s2s.latency_ms`: caller stopped to first
synthesized audio. "Caller stopped" is Coval's `utterance_end` marker when it
sent one, otherwise our `vad_done`, and `s2s.latency.source` says which.

Known caveat: Coval currently sends `utterance_end` about 350 ms after the
persona actually stops, so the headline reads about 350 ms low on every stack.
Comparisons between stacks are fine; absolute numbers are not yet.

## Adding things

A **stack** is a JSON file in `src/agent/stacks/` naming a provider and model
per role. If the provider is new, add its Pipecat class to the table in
`src/agent/services/` and its extra to `pyproject.toml`.

A **scenario** is a directory in `src/agent/scenarios/`. See the README there.

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src tests
uv run pytest -q --disable-socket --allow-unix-socket --allow-hosts=127.0.0.1,::1
```
