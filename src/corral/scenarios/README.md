# Scenarios

A scenario is a directory here with:

- `system-prompt.txt`: the prompt the agent runs
- `summary.txt`: its one line in the menu
- `coval.json`: the Coval suite it's benchmarked with. Leave this out for a
  talk-only scenario.
- `tools.py`: the tools the agent can call, if it has any. Leave this out and
  the agent just talks.

`default` is a plain assistant for trying models by voice.
`id_roulette` is one short call: the caller reads out a detail once and the
agent looks it up as it heard it. The details are account ids, emails, card
numbers, postal codes and spelled surnames, three of each. Its personas run
from a quiet room to an office to a landline with an air conditioner, so it
shows which details and which caller conditions make agents mishear.

## tools.py

`tools.py` has one function, `environment()`, that returns a fake backend for
one call: a list of tools, `start(test_case)` to seed it, `finish()` to clean
up. Every call gets its own, so calls can't see each other's bookings.

The easy version is canned answers from a `tools.json` next to it, like
`id_roulette` has: each tool answers the same thing however it's called, and
a test case swaps in different answers. When you need answers that depend on the
arguments, or state that changes after a write, write your own in `tools.py`
with those same three parts. Handlers take the arguments as a dict and return
the result. They never see Pipecat.

A tool in `tools.json` can also check what the agent passed it. List the check
by name with the right value for each argument: one string for every test case,
or a map from test case to its string, the way `id_roulette`'s `lookup` has
`"character_error_rate": {"value": {"email-1": "j.okafor_88@example.com", ...}}`.
When the call ends, the agent's first call of that tool is checked against the
test case's value. Case and spaces don't count, and punctuation only counts
where the right value has it, so an email's dots and @ matter but dashes in a
card number don't. The result goes to Coval on a span named after the check,
with one attribute per `<tool>.<argument>`, so `lookup.value` on a
`character_error_rate` span. Read it with a custom trace metric in `coval.json`,
the way `id_roulette` does.

To seed the backend differently per test case, give each case in `coval.json`
a `test_case` naming the seed. When Coval calls, the agent asks Coval which
test case is on the line and seeds the backend for it, so the Coval key needs
`simulations:read`. A name the backend doesn't know fails before the run
starts.

## coval.json

**With IDs**
```json
{
  "workspace": "EXAMPLE_WORKSPACE_ID_123",
  "test_set": "EXAMPLE_TESTSET_ID_123",
  "personas": ["EXAMPLE_PERSONA_ID_123", "EXAMPLE_PERSONA_ID_456"],
  "metrics": ["EXAMPLE_METRIC_ID_123", "EXAMPLE_METRIC_ID_456"]
}
```

**With Full Spec**
```json
{
  "workspace": "EXAMPLE_WORKSPACE_ID_123",
  "test_set": {
    "display_name": "My suite",
    "test_cases": [
      {
        "input_str": "Ask for a dinner idea.",
        "expected_behaviors": ["Suggests a dish"]
      }
    ]
  },
  "personas": [
    {
      "name": "Easy caller",
      "persona_prompt": "Speak clearly and stay on task.",
      "voice_name": "aria",
      "language_code": "en-US"
    }
  ],
  "metrics": [
    {
      "metric_name": "Answered the request",
      "description": "Did the agent give the caller what they asked for?",
      "metric_type": "METRIC_LLM_BINARY",
      "prompt": "Did the agent answer every request the caller made?"
    }
  ]
}
```

You can mix the two. Use an id for anything you've already built in Coval,
and write out the rest. A written-out one is just what you'd send Coval's API
to create it, plus `test_cases` on the test set.

Corral makes the written-out ones in your workspace on the first run. After
that it keeps them in sync with the file, so edit the file, not Coval. It
finds them by name, and test cases by `input_str`, so anything in the
workspace with the same name gets updated to match. Delete a test case from
the file and it's gone from Coval too.

Every persona gets its own run.

To keep many test cases but place only some per run, set `sample` to how many.
Corral asks Coval for that many, picked by the UTC date, so every agent and
persona benchmarked the same day gets the same ones, and the set rotates from
day to day. `id_roulette` uses `"sample": 1`. `--smoke` still picks one at
random.

`workspace` is where all of it lives. If you leave it out, Corral uses
`CORRAL_WORKSPACE_ID` from your `.env`. If that's empty too, it finds the
workspace called Corral, or makes it, and saves the id to your `.env`.

**With assets**
```json
{ "assets": "workflow-bank.v1", "personas": ["clean"], "metrics": ["judge", "latency"] }
```
Instead of ids, name a `<scenario>.<version>` in a manifest shaped like the
scenario map `benchmark-infra/assets` publishes, and set `CORRAL_ASSETS_URI`
to it: `gs://coval-bench-asset-versions/coval-benchmarking/scenario_map.json`
for the production assets, or a local file with the same shape for another
org. The workspace, test set, personas and metrics come from that entry.
`personas` are its conditions and `metrics` its metric names; leave either out
for all of them. A gs:// manifest is read with your Google login
(`gcloud auth application-default login`) and needs
`roles/storage.objectViewer` on the bucket.

To benchmark your own agent, add a directory with a prompt, a summary and
a `coval.json`. It shows up in the menu on its own.
