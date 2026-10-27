# Security Policy

## Reporting a vulnerability

Report suspected vulnerabilities privately with GitHub's **Report a vulnerability** form in this repository's Security tab. Do not open a public issue or pull request for a security report.

Include the commit you ran, the impact you observed, and the smallest reproduction that does not expose API keys, recordings, transcripts or customer data. Coval will acknowledge the report through the private advisory and coordinate any disclosure there.

## Handling keys

Corral reads Coval and model-provider keys from `.env`, which is gitignored. Keep real keys there or in your own secret store, never in a scenario, a test or an issue.
