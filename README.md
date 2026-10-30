# Corral

Corral is how Coval wrangles the voice AI models we benchmark. You can use it
to quickly spin up and talk to native or cascade voice agents on your local 
machine or, if you have a Coval account, automatically eval them with Coval.

Local runs are great for hearing how the model sounds and feels, getting a 
sense for its latency, and iterating quickly. Coval evals are great for 
formalizing and automating checks for the issues that you come across in your
local run.

We recommend using corral exactly like we do at Coval. Choose a few different
model stacks to try with your agent and talk with them on your local machine.
Write down all the things that bother you or that you find fishy. Create tests
in Coval that uncover these things. When you try out your next batch of agents,
these Coval tests will automate the intuition you developed, freeing you up to
discover entirely new ways that agents tend to break down.

The flow is: 
1. Test Manually
2. Automate
3. Monitor 
4. Repeat

Every cycle your evals will become better and better about identifying issues 
as they appear in your agent.

## Setup

To set up Corral use the following commands. Don't forget to fill in the .env
with API keys for any models that you plan to use.

```bash
uv sync
cp .env.example .env    # then fill in your keys
```

## Run

```bash
uv run corral
```

The CLI menu walks you through picking a mode, a scenario and the models. The
agent is a cascade, an STT, an LLM and a TTS in a line, or one native
speech-to-speech model that takes audio in and gives audio out. The menu asks
which when you have a key for a native model.

- **talk** Spins up the agent locally letting you talk with it through your mic
  and speakers. Use Ctl+c to quit. If the scenario has tools, the menu asks
  which test case to seed them for (`--test-case` when scripting), and the
  tools called print after the latency table.
- **benchmark** Runs your eval suite on Coval against the agent and then prints
  the results (you can also see them in the Coval app). The agent dials in to
  Coval for each call through the Coval Agent SDK, so nothing on your machine
  needs a public address. Use `--smoke` to run a single test case.
- **check-access** Probes every API: one authenticated request to Coval and to
  each vendor with a key in `.env`, then a session opened on each native
  model's own socket, so a key that has died or a model its vendor has stopped
  serving shows up here rather than on a call. Vendors without a key are
  skipped. Exits non-zero if anything with a key fails, so it can run daily or
  ahead of a benchmark in CI.

The Docker image runs `corral benchmark` by default. Its entrypoint is `corral`,
so arguments you pass replace `benchmark`, as in `benchmark --s2s
openai/gpt-realtime-2.1`. The scenario and models can also come from the `S2S_*`
environment variables the flags default to.

For scripts, pass the mode and flags instead of answering prompts:


```bash
uv run corral benchmark --scenario id_roulette --stt soniox/stt-rt-v5 --llm google/gemini-2.5-flash --smoke
```

A native model is `--s2s` on its own, with no `--stt`, `--llm` or `--tts`:

```bash
uv run corral talk --s2s openai/gpt-realtime-2.1
```

To run a scenario from the benchmark assets rather than a scenario directory,
name it with `--assets` and set `CORRAL_ASSETS_URI` to the manifest. The test
set, personas, metrics and workspace come from that entry, and the system
prompt from the entry's agent for your model, as that agent is saved in Coval.
When your model isn't one of the scenario's agents, `--prompt-from` names the
one whose prompt to use. `--personas` and `--metrics` take comma-separated
names from the entry; leave them out for all of them. The production manifest
needs the production Coval key.

```bash
CORRAL_ASSETS_URI=gs://coval-bench-asset-versions/coval-benchmarking/scenario_map.json \
  uv run corral benchmark --assets workflow-bank.v1 --personas clean \
  --s2s openai/gpt-realtime-2.1 --prompt-from openai/gpt-realtime-2
```

Gemini 3.8 Live is named as it is in the map, so it needs no `--prompt-from`;
its session settings in `s2s.json` match its Coval agent.

Gemini 3.8 Live Extended Thinking is named as it is in the map, so it needs no
`--prompt-from`. Its session settings match 3.8 Live's Coval agent, plus a `MEDIUM`
thinking level, which the Coval agent can't set.

Grok Voice 3.0 is named as it is in the map, so it needs no `--prompt-from`; its
session settings in `s2s.json` match its Coval agent. It needs `XAI_API_KEY`.

Eleven Realtime v1 Mini and Max are named as they are in the map, so they need no
`--prompt-from`; their session settings in `s2s.json` match their Coval agents.

## What gets measured

Every answered turn is a `turn_latency` span in Coval with `stt`, `llm` and
`tts` under it, each tagged with the provider and model that ran.

- `turn_latency_ms` is the headline. Caller's last word to the first audio the
  agent plays back. Last word is Silero firing minus its silence window, on
  our clock, so it means the same thing locally and on Coval.
- `stt_ms` is our finalize request to the STT's confirmation.
- `llm_ms` is the request to the first token.
- `tts_ms` is the first sentence to the first audio.
- `stt_timed_out` is true when the STT never answered the finalize. That turn
  has no `stt_ms`.
- `silero_missed` is true when Silero never heard the caller and smart-turn
  ended the turn on its own. Nothing asked the STT to finalize, so that turn
  has no `stt_ms` either, and the mic level is the first thing to check.
- `smart_turn_timed_out` is true when smart-turn couldn't decide and the
  silence fallback ended the turn.

The stages don't add up to the headline. The rest is turn-taking, which is our
code and the same for every stack.

When the model calls a tool, the turn makes two LLM requests with the tool in
between, so `tool_calls` counts the calls and the LLM time is split: `llm_ms`
ends at the tool call, `tool_ms` is the handler running, and `llm_second_ms`
is the request after the result up to its first token. Each call is also an
`llm_tool_call` span with the function's name, arguments and result, which is
what Coval's tool call metric reads. A model that says "one moment" and then
calls the tool has already ended the turn, so that call is a span on its own
and not in the turn's count.
When a scenario's tools check their arguments, the call also ends with a
`character_error_rate` span holding each checked argument's rate.

Every call is also one `call` span from the moment the agent picks up to the
hang-up, so a call that drags shows up in its length.

A native model does its own turn-taking, so its turn is measured differently
under the same headline. Silero still runs on the caller's audio, but only as a
stopwatch for the last word; it has no say in the turn. The span has
`endpointing` and `model` under it.

- `turn_latency_ms` means the same thing: last word to first audio. For a
  cascade that gap includes our turn-taking; for a native model the vendor's.
- `endpointing_ms` is the last word to the moment the vendor said the caller
  was done. Negative means the model answered before Silero thought the caller
  had finished. A vendor that never says has none. gpt-live-1 listens while it
  talks and never says, and neither does Gemini Live, so their `model_ms` starts
  at the last word.
- `model_ms` is from there to the first audio. A tool call splits it into
  `model_ms`, `tool_ms` and `model_second_ms`.
- `silero_missed` still applies; `stt_timed_out` and `smart_turn_timed_out`
  don't. gpt-live-1 and Gemini Live have nothing else to start a turn from, so a
  turn Silero missed isn't counted.

The last word is a guess. Silero fires on a mid-sentence pause too, and nothing
cleans that up for a native model the way smart-turn does for a cascade.

## Adding things

To add a model, put it in `src/corral/models/stt.json`, `llm.json`, `tts.json`
or, for a native speech-to-speech model, `s2s.json`. The first one in the file
is the default. A native row's voice and turn detection are vendor knobs, so
they go in its `options` the way that vendor's Pipecat settings want them. A
vendor with more than one speech-to-speech API takes `api` to pick one, and a
live model's `delegation` is the backend model it hands its reasoning to. If
it's a provider we don't have yet, give it
an entry in `src/corral/models/__init__.py` with its key and Pipecat class and
add its extra to `pyproject.toml`.

To add a scenario, make a directory in `src/corral/scenarios/` with a system
prompt and, if you want to benchmark it, a Coval suite. If the agent should be
able to call tools, add a `tools.py` with the fake backend they act on. The
README in there has the details.

## Checks

These are the same checks CI runs:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src tests
uv run pytest
```
