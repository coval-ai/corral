# Scenarios

A scenario is a directory here with:

- `system-prompt.txt`: the prompt the agent runs
- `summary.txt`: its one line in the menu
- `coval.json`: the Coval suite it's benchmarked with. Leave this out for a
  talk-only scenario.
- `tools.py`: the tools the agent can call, if it has any. Leave this out and
  the agent just talks.

`default` is a plain assistant for trying models by voice. `ultrabank` is
Ultra Bank with the short daily test set.

## tools.py

`tools.py` has one function, `environment()`, that returns a fake backend for
one call: a list of tools, `start(test_case)` to seed it, `finish()` to clean
up. Every call gets its own, so calls can't see each other's bookings.

The easy version is canned answers from a `tools.json` next to it, loaded
with `JsonEnvironment.load`: each tool answers the same thing however it's
called, and a test case swaps in different answers. When you need answers that depend on the
arguments, or state that changes after a write, write your own in `tools.py`
with those same three parts. Handlers take the arguments as a dict and return
the result. They never see Pipecat.

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

`workspace` is where all of it lives. If you leave it out, Corral uses
`CORRAL_WORKSPACE_ID` from your `.env`. If that's empty too, it finds the
workspace called Corral, or makes it, and saves the id to your `.env`.

To benchmark your own agent, add a directory with a prompt, a summary and
a `coval.json`. It shows up in the menu on its own.
