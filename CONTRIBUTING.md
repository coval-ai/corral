# Contributing

This repository is public so anyone can run the same voice agents Coval benchmarks and see exactly how the numbers are measured. Coval maintains the implementation and does not accept pull requests from outside the Coval organization.

If you find a bug, a model that misbehaves, or a measurement that looks wrong, open an issue with the command you ran, the models you picked and what you expected to see. Leave out API keys, recordings, transcripts and anything else you would not post publicly. Coval will triage the report and make accepted changes internally.

Report suspected vulnerabilities privately as described in [SECURITY.md](SECURITY.md), not in an issue.

Automated dependency updates from Dependabot are the only outside pull requests the repository accepts.

## Working on Corral at Coval

The [README](README.md) covers setup, running Corral and the checks CI runs. Start a branch from `main`, run `uv run pre-commit install` once so the generated workflow keeps up with the catalogs, run those checks before you push, and keep the tests offline: they block the network outside localhost, so no test calls Coval or a model provider.
