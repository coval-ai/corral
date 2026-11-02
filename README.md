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

The menu will ask you for a mode, a scenario and which models to use. You can 
run the agent as either a cascade (an STT, an LLM and a TTS chained together) 
or a single native speech-to-speech model that takes audio in and gives audio 
back.

- **talk** puts the agent on your mic and speakers so you can have a
  conversation with it. Ctrl+C hangs up.
- **benchmark** runs the scenario's Coval suite against the agent and prints
  the scores when Coval is done. The agent dials in to Coval using the Coval
  Agent SDK, so you don't need to expose anything on your machine.
- **check-access** makes one request to Coval, to each vendor and to each
  model you have a key for to make sure that everything's still valid. Good 
  first step when debugging.

You can also run corral as a script. Here are some examples:

```bash
uv run corral talk --s2s openai/gpt-realtime-2.1
uv run corral benchmark --scenario id_roulette --stt soniox/stt-rt-v5 --llm google/gemini-2.5-flash --smoke
```

- `--stt`, `--llm`, `--tts` pick the cascade models. Each defaults to the
  first row in its catalog file.
- `--s2s` picks a native model instead. Don't pass the three cascade flags
  with it.
- `--scenario` picks the scenario directory.
- `--test-case` (talk) seeds the scenario's tools for one test case.
- `--smoke` (benchmark) runs a single test case, handy for checking the wiring.
- `--sample N` (benchmark) runs N test cases, the same N for the rest of the
  day so different models get the same ones.
- `--personas` (benchmark) names which of the suite's personas to run, or
  `all`. Default is the first.
- `--assets`, `--metrics`, `--prompt-from` (benchmark) run a scenario from the
  benchmark assets manifest at `CORRAL_ASSETS_URI` instead of a directory.

`corral --help` lists these. Each mode's `--help` lists the models you have keys for.

The Docker image runs `corral benchmark` by default, and any arguments you
pass replace that.

## Adding things

To add a model, add a row to `src/corral/models/stt.json`, `llm.json`,
`tts.json` or, for a native model, `s2s.json`. The first row in each file is
the default. If it's a provider we don't support yet, it also needs an entry in
`src/corral/models/__init__.py` with its key and Pipecat class, plus its extra
in `pyproject.toml`.

To add a scenario, make a directory in `src/corral/scenarios/` with a system
prompt. Add a Coval suite if you want to benchmark it, and a `tools.py` if the
agent should be able to call tools. [The README in there](src/corral/scenarios/README.md)
walks through it.

## Checks

These are the same checks CI runs:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src tests
uv run pytest
```
