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

The CLI menu walks you through picking a mode, a scenario and the models.

- **talk** Spins up the agent locally letting you talk with it through your mic
  and speakers. Use Ctl+c to quit. If the scenario has tools, `--test-case`
  picks which seed they answer from, and the tools called print after the
  latency table.
- **benchmark** Serves the agent to Coval which runs your eval suite against it
  and then prints the results (you can also see them in the Coval app). Use 
  `--smoke` to a single test case. The benchmark command needs `brew install 
  cloudflared`.
- **serve** is what the Docker image runs. It puts the agent on port 7873.
  Use `--port` or set `PORT` to run it on a different port.

For scripts, pass the mode and flags instead of answering prompts:


```bash
uv run corral benchmark --scenario ultrabank --stt soniox/stt-rt-v5 --llm google/gemini-2.5-flash --smoke
```

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
- `smart_turn_timed_out` is true when smart-turn couldn't decide and the
  silence fallback ended the turn.

The stages don't add up to the headline. The rest is turn-taking, which is our
code and the same for every stack.

## Adding things

To add a model, put it in `src/corral/models/stt.json`, `llm.json` or
`tts.json`. The first one in the file is the default. If it's a provider we
don't have yet, give it an entry in `src/corral/models/__init__.py` with its
key and Pipecat class and add its extra to `pyproject.toml`.

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
