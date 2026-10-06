# Scenarios

A scenario is a directory here with:

- `system-prompt.txt`: the prompt the agent runs
- `summary.txt`: its one line in the menu
- `coval.json`: the Coval test set, persona and metric ids, if it can be
  benchmarked. Keys starting with `_` are the human names. Leave this out
  for a talk-only scenario.

A `system-prompt.<stack>.txt` next to the default is used for that stack
instead. The stack digest covers whichever prompt ran.

`default` is a plain assistant for trying a stack by voice. `baseline` is
Ultra Bank with the short daily test set.
