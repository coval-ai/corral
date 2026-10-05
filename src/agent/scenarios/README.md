# Scenarios

One directory per scenario, and a scenario is one prompt plus, optionally,
one Coval test set. `summary.txt` is its line in the menu. `system-prompt.txt`
is the default prompt; a `system-prompt.<stack>.txt` beside it is used instead
for that stack, and the stack digest covers whichever one ran. `coval.json`
names the test set, the persona that calls it and the metrics that score it,
with `_`-prefixed keys for the human names; a scenario without it can be
talked to but not benchmarked. The ids are public, so anyone with a Coval key
can run a scenario.

`default` is a generic assistant for trying a stack. `baseline` is Ultra Bank
with the short daily set.
