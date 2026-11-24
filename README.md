# Corral

Corral is how Coval wrangles the voice AI models we benchmark. You can use it
to quickly spin up and talk to native or cascade voice agents on your local 
machine or, if you have a Coval account, automatically eval them with Coval.

https://github.com/user-attachments/assets/11d84436-9280-458e-a19a-1c98d0ec7659

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
  conversation with it. Each answered turn logs one line with its latency, and
  Ctrl+C hangs up and prints the table. `LOGURU_LEVEL=DEBUG` shows Pipecat's
  full log.
- **benchmark** runs the scenario's Coval suite against the agent and prints
  the scores when Coval is done. The agent dials in to Coval using the Coval
  Agent SDK, so you don't need to expose anything on your machine.
- **check-access** makes one request to Coval, to each vendor and to each
  model you have a key for to make sure that everything's still valid. Good 
  first step when debugging.

You can also run corral as a script. Here are some examples:

```bash
uv run corral talk --s2s openai/gpt-realtime-2.1
uv run corral benchmark --scenario id_roulette --stt soniox/stt-rt-v6 --llm google/gemini-2.5-flash --smoke
```

- `--stt`, `--llm`, `--tts` pick the cascade models. Each defaults to the
  first row in its catalog file.
- `--s2s` picks a native model instead. Don't pass the three cascade flags
  with it.
- `--scenario` picks the scenario directory.
- `--scenario-path` runs a scenario directory kept outside Corral instead,
  like `--scenario-path ./my-scenario`.
- `--test-case` (talk) seeds the scenario's tools for one test case.
- `--smoke` (benchmark) runs a single test case, handy for checking the wiring.
- `--test-cases` (benchmark) runs just the test cases you name, like
  `--test-cases email-1,postcode-2`, and `--repeats` runs each one that many times.
- `--json` (benchmark) is for running Corral from another program. See
  [Running it from a script](#running-it-from-a-script).
- `--full` (benchmark) runs every test case even when the suite's `coval.json`
  samples by default.
- `--sample N` (benchmark) runs N test cases, the same N for the rest of the
  day so different models get the same ones.
- `--personas` (benchmark) names which of the suite's personas to run, or
  `all`. Default is the first.
- `--assets`, `--metrics`, `--prompt-from` (benchmark) run a scenario from the
  benchmark assets manifest at `CORRAL_ASSETS_URI` instead of a directory.

Set `CORRAL_MODELS_URI` to a JSON list, as `gs://bucket/object` or a local path, to offer
models that aren't in the catalog. A vendor in it names a service Corral already runs
(`speaks`), its `base_url` and the variable its key is read from (`key_env`).

`corral --help` lists these. Each mode's `--help` lists the models you have keys for.

The Docker image runs `corral benchmark` by default, and any arguments you
pass replace that.

On GitHub, the Benchmark action runs the same thing on Cloud Run: pick a
scenario, a stack and a size from the dropdowns under Actions. The dropdowns
come from the model catalogs, the scenario directories and
`.github/benchmarks.json`. The Daily action runs every row of the `daily` list
in that file each night at 09:00 UTC, one Benchmark run per model and scenario,
and you can run it by hand for a rerun. A pre-commit hook regenerates both
workflows when you commit a change to any of those and stops the commit so you
can stage it; set it up once per clone with `uv run pre-commit install`. CI
checks the committed workflows are current.

## Running it from a script

At a terminal a benchmark prints a summary you can read and stays for Coval's
scores. A program that runs Corral wants the ids instead, so it can fetch the
scores itself. Add `--json`:

```bash
uv run corral benchmark --scenario id_roulette --s2s openai/gpt-realtime-2.1 --test-cases email-1,postcode-2 --repeats 2 --json
```

Stdout is then one JSON document and nothing else, and the log stays on
stderr. Corral leaves as soon as the calls are over, without waiting for
Coval to score them.

```json
{
  "agent": "openai-gpt-realtime-2.1",
  "digest": "9f2c...",
  "runs": [
    {
      "run_id": "RUN_ID",
      "persona": "ID Roulette: quiet room",
      "simulations": [
        {"simulation_id": "SIM_1", "test_case": "email-1", "repeat": 1, "status": "completed", "error": null},
        {"simulation_id": "SIM_2", "test_case": "email-1", "repeat": 2, "status": "failed", "error": "why the call failed"}
      ]
    }
  ]
}
```

There is one run per persona, and one simulation per test case per repeat. A
simulation's `status` says whether the call happened, not how the agent did;
the scores are in Coval under its `simulation_id`. The exit code is 0 whenever
the document was printed, failed calls included, and anything else means there
is no document to read.

## Adding things

To add a model, add a row to `src/corral/models/stt.json`, `llm.json`,
`tts.json` or, for a native model, `s2s.json`. The first row in each file is
the default. If it's a provider we don't support yet, it also needs an entry in
`src/corral/models/__init__.py` with its key and Pipecat class, plus its extra
in `pyproject.toml`.

To add a scenario, make a directory in `src/corral/scenarios/` with a system
prompt. Add a Coval suite if you want to benchmark it, and a `tools.json` if the
agent should be able to call tools. [The README in there](src/corral/scenarios/README.md)
walks through it.

## Checks

These are the same checks CI runs:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy --strict src tests scripts
uv run pytest
```
