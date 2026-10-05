# Scenarios

One directory per scenario. `default` is a generic assistant for talking to a
stack; `bank` is what the benchmark runs. `system-prompt.txt` is the default prompt; a
`system-prompt.<stack>.txt` beside it is used instead for that stack, and the
stack digest covers whichever one ran. `test-sets.json` names the scenario's
test sets: each entry is a Coval test set id, the persona that calls it and the
metrics that score it, with `_`-prefixed keys for the human names. The ids are
public, so a default set can be run by anyone with a Coval key.
