# Corral

Coval's own cascade voice agent for the S2S leaderboard. Corral rounds up a
speech-to-text, LLM and text-to-speech model into one Pipecat agent, lets
Coval's simulator call it, and times every turn from inside the pipeline so
we know which model the pause came from.

## Setup

```bash
uv sync
cp .env.example .env    # then fill in your keys
```

## Run

```bash
uv run corral
```

The TUI menu walks you through picking a mode, a scenario and the models.

- **talk** puts the agent on your mic and speakers. Ctrl-C hangs up and
  prints a latency table. Needs `brew install portaudio && uv sync --extra local`.
- **benchmark** serves the agent behind a Cloudflare tunnel, runs one Coval
  benchmark against it, prints the metrics and exits. The tunnel takes about
  80 seconds to come up. `--smoke` runs a single test case. Needs
  `brew install cloudflared`.
- **serve** is what the Docker image runs: the agent on a port, nothing else.

For scripts, pass the mode and flags instead of answering prompts:

```bash
uv run corral benchmark --scenario ultrabank --stt soniox/stt-rt-v5 --llm google/gemini-2.5-flash --smoke
```

Leave a model out and you get the first one in its list, or whatever
`S2S_STT`, `S2S_LLM`, `S2S_TTS` and `S2S_SCENARIO` are set to.

The menu marks every model whose key is missing from `.env` and defaults to
the first one that has a key. Each pick sends one request to its provider
before the next question, and a provider that answers anything but 200 is
shown with the code so you can pick another. A scripted run makes the same
three checks before anything else, so a bad key fails before the tunnel.

## What gets measured

Every answered turn becomes a `turn_latency` span in Coval with `stt`, `llm`
and `tts` spans under it, each tagged with the provider and model that ran.

The headline, `s2s.turn_latency_ms`, is from the caller's last word to the
first synthesized audio. "Last word" is Silero firing minus its silence
window, on our clock, so it means the same thing locally and on Coval.

Each stage is the provider's own request and nothing else: `stt_ms` is our
finalize request to the STT's confirmation, `llm_ms` is the request to the
first token, `tts_ms` is the first sentence to the first audio. The stages
don't add up to the headline. The rest is turn-taking, which is ours and the
same for every stack.

Two flags call out turns that need a second look. `stt_confirmed` is false
when the STT never answered the finalize, and that turn has no `stt_ms`.
`turn_detection_timed_out` is true when smart-turn couldn't decide and its
silence fallback ended the turn.

## Adding things

A **model** is an entry in `src/corral/models/stt.json`, `llm.json` or
`tts.json`. First in the file is the default. A new provider also needs its
Pipecat class in `src/corral/models/__init__.py` and its extra in `pyproject.toml`.

A **scenario** is a directory in `src/corral/scenarios/`. See the README there.

## Checks

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src tests
uv run pytest
```
