# Corral

Coval's own voice agent for the S2S leaderboard, and the harness around it.
Corral rounds up a speech-to-text, LLM and text-to-speech model into one
Pipecat agent, lets Coval's simulator call it, and times every turn from
inside the pipeline so we know which model the pause came from.

## Setup

```bash
uv sync
cp .env.example .env    # then fill in your keys
```

## Run

```bash
uv run corral
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
uv run corral benchmark --scenario baseline --stt soniox/stt-rt-v5 --llm google/gemini-2.5-flash --smoke
```

Leave a modality out and it gets the first model on the list. `S2S_STT`, `S2S_LLM`,
`S2S_TTS` and `S2S_SCENARIO` set the defaults. Benchmarks need
`COVAL_API_KEY`; talk needs the provider keys for the models you picked. If
something is missing, the menu says so and gives the command.

## What gets measured

Every answered turn becomes a `turn_latency` span in Coval, with `stt`, `llm`
and `tts` spans under it showing how long each provider took and which model
ran. We grab the time at these moments:

- `vad_stop`: Silero hears enough silence and Pipecat asks the STT to finalize
- `stt_done`: the STT confirms the transcript is final
- `vad_done`: smart-turn agrees the caller is done
- `llm_start`: the LLM request goes out
- `llm_first_token`: the model says its first word
- `tts_start`: the first text goes to the TTS
- `tts_first_audio`: we have the first bit of synthesized audio
- `audio_start`: we send the first audio chunk back to Coval

Each one is on the span as `s2s.<moment>.unix_ms`. The headline is
`s2s.turn_latency_ms`: caller stopped to first synthesized audio. "Caller
stopped" is when Silero fired minus its silence window, which is when the
caller's last word reached us. It's our clock, so it means the same thing
locally and on Coval, and the same thing for every stack.
When smart-turn never decides and its silence fallback ends the turn,
`s2s.turn_latency.turn_detection_timed_out` is true.

Each stage is measured at its own request boundary, so it's that provider's
time and nothing else: `s2s.turn_latency.stt_ms` is Silero firing to the STT
confirming the transcript, `llm_ms` is the request going out to the first
token, `tts_ms` is text going to the TTS to the first audio. Pipecat times the
LLM and TTS requests itself. Every STT in the catalog runs with its own
endpointing off so they're all measured the same way, and each one has to
confirm our finalize request: that confirmation is the only thing that ends the
STT stage. A turn where it never came has no `stt_ms` and
`s2s.turn_latency.stt_confirmed` false, so a provider that drops finalizes shows
up as that rather than as a slow one. The stages don't add up
to the headline; the rest is turn-taking and plumbing, which is ours.

## Adding things

A **model** is an entry in `src/corral/models/stt.json`, `llm.json` or `tts.json`.
The first in each file is the default. If the provider is new, add its Pipecat class to the
table in `src/corral/models/__init__.py` and its extra to `pyproject.toml`.

A **scenario** is a directory in `src/corral/scenarios/`. See the README there.

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src tests
uv run pytest -q --disable-socket --allow-unix-socket --allow-hosts=127.0.0.1,::1
```
