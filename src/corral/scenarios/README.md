# Scenarios

A scenario is a directory here with:

- `system-prompt.txt`: the prompt the agent runs
- `summary.txt`: its one line in the menu
- `coval.json`: the Coval test set, persona and metric ids, if it can be
  benchmarked. Leave this out for a talk-only scenario.

`default` is a plain assistant for trying models by voice. `baseline` is
Ultra Bank with the short daily test set.
