# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""Write .github/workflows/benchmark.yml and daily.yml from the repo.

The scenario options are the manifest scenarios listed in .github/benchmarks.json, each with the
agent whose prompt a model not in the manifest borrows, plus every directory under
src/corral/scenarios with a coval.json. The stack options are the native models
in models/s2s.json plus the cascades listed in benchmarks.json. The daily rows in
benchmarks.json each name a scenario, the stacks to run it on (``s2s``, ``cascades`` or a stack
option), a size and personas; daily.yml runs every model and scenario pair on a schedule. Run it
after changing any of those; ``--check`` exits 1 when a committed workflow is stale, which is
what CI does.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "src" / "corral" / "models"
SCENARIOS = ROOT / "src" / "corral" / "scenarios"
TRACKED = ROOT / ".github" / "benchmarks.json"
WORKFLOWS = ROOT / ".github" / "workflows"

REPO_SUFFIX = " (repo)"
CASCADE_JOIN = " + "
SIZES = ["smoke", "sample 5", "full"]
PERSONAS = ["default", "all"]
SELECTORS = ("s2s", "cascades")
DAILY_CRON = "0 9 * * *"
DAILY_PARALLEL = 2


def model_ids(modality: str) -> list[str]:
    rows = json.loads((MODELS / f"{modality}.json").read_text())
    return [f"{row['provider']}/{row['name']}" for row in rows]


def scenario_options(assets: dict[str, str | None]) -> list[str]:
    dirs = sorted(p.name for p in SCENARIOS.iterdir() if (p / "coval.json").is_file())
    return list(assets) + [name + REPO_SUFFIX for name in dirs]


def prompt_cases(assets: dict[str, str | None]) -> str:
    """A scenario whose assets carry their own prompt file borrows no agent's prompt."""
    indent = " " * 12
    return "\n".join(
        f'{indent}"{name}") add --prompt-from "{agent}" ;;'
        for name, agent in assets.items()
        if agent
    )


def stack_options(tracked: list[list[str]]) -> list[str]:
    offered = {m: set(model_ids(m)) for m in ("stt", "llm", "tts")}
    cascades = []
    for stt, llm, tts in tracked:
        for modality, pick in (("stt", stt), ("llm", llm), ("tts", tts)):
            if pick not in offered[modality]:
                raise SystemExit(f"{TRACKED}: {pick} isn't in models/{modality}.json")
        cascades.append(CASCADE_JOIN.join((stt, llm, tts)))
    return model_ids("s2s") + cascades


def inputs_block(indent: str, scenarios: list[str], stacks: list[str], *, choice: bool) -> str:
    def field(name: str, description: str, options: list[str]) -> str:
        lines = [f"{indent}{name}:", f"{indent}  description: {description}"]
        if choice:
            lines.append(f"{indent}  type: choice")
            lines.append(f"{indent}  options: {json.dumps(options)}")
        else:
            lines.append(f"{indent}  type: string")
        lines.append(f"{indent}  default: {json.dumps(options[0])}")
        return "\n".join(lines)

    return "\n".join(
        [
            field("scenario", "Coval scenario, or a scenario directory in the repo", scenarios),
            field("stack", "One native model, or a cascade as STT + LLM + TTS", stacks),
            field("size", "How much of the suite to run", SIZES),
            field(
                "personas",
                "The suite's first persona, all of them, or manifest persona keys joined by commas",
                PERSONAS,
            ),
        ]
    )


def daily_units(
    rows: list[dict[str, Any]], scenarios: list[str], stacks: list[str]
) -> list[dict[str, str]]:
    """One matrix entry per model and scenario the daily rows ask for, in row order."""
    natives = model_ids("s2s")
    cascades = [s for s in stacks if CASCADE_JOIN in s]
    units = []
    for row in rows:
        scenario, size = str(row["scenario"]), str(row["size"])
        personas = str(row.get("personas", PERSONAS[0]))
        if scenario not in scenarios:
            raise SystemExit(f"{TRACKED}: daily scenario {scenario} isn't a scenario option")
        if size not in SIZES and not (size.startswith("sample ") and size[7:].isdigit()):
            raise SystemExit(f"{TRACKED}: daily size must be smoke, sample N or full: {size}")
        if not re.fullmatch(r"[A-Za-z0-9._/,-]+", personas):
            raise SystemExit(f"{TRACKED}: daily personas must be keys joined by commas: {personas}")
        for pick in row["stacks"]:
            if pick == "s2s":
                chosen = natives
            elif pick == "cascades":
                chosen = cascades
            elif pick in stacks:
                chosen = [str(pick)]
            else:
                raise SystemExit(f"{TRACKED}: daily stack {pick} isn't s2s, cascades or a stack")
            for stack in chosen:
                units.append(
                    {"scenario": scenario, "stack": stack, "size": size, "personas": personas}
                )
    return units


def matrix_block(indent: str, units: list[dict[str, str]]) -> str:
    return "\n".join(f"{indent}- {json.dumps(unit)}" for unit in units)


def render() -> dict[Path, str]:
    tracked: dict[str, Any] = json.loads(TRACKED.read_text())
    assets = {str(name): entry.get("prompt_from") for name, entry in tracked["assets"].items()}
    scenarios = scenario_options(assets)
    stacks = stack_options([[str(part) for part in cascade] for cascade in tracked["cascades"]])
    dispatch = inputs_block("      ", scenarios, stacks, choice=True)
    call = inputs_block("      ", scenarios, stacks, choice=False)
    benchmark = TEMPLATE.format(
        dispatch=dispatch,
        call=call,
        repo_suffix=REPO_SUFFIX,
        prompt_cases=prompt_cases(assets),
    )
    units = daily_units(list(tracked.get("daily", [])), scenarios, stacks)
    daily = DAILY_TEMPLATE.format(
        cron=DAILY_CRON,
        parallel=DAILY_PARALLEL,
        matrix=matrix_block("          ", units),
    )
    return {WORKFLOWS / "benchmark.yml": benchmark, WORKFLOWS / "daily.yml": daily}


TEMPLATE = """\
# Generated by scripts/gen_benchmark_workflow.py from the model catalogs, the scenario
# directories and .github/benchmarks.json. Edit those and commit; a pre-commit hook rewrites
# this file and CI checks it's current.

name: Benchmark
run-name: ${{{{ inputs.scenario }}}} · ${{{{ inputs.stack }}}} · ${{{{ inputs.size }}}}

on:
  workflow_dispatch:
    inputs:
{dispatch}
  workflow_call:
    inputs:
{call}

permissions:
  contents: read
  id-token: write

jobs:
  benchmark:
    runs-on: ubuntu-latest
    environment: gcp
    timeout-minutes: 130
    steps:
      - name: Turn the picks into corral flags
        id: flags
        env:
          SCENARIO: ${{{{ inputs.scenario }}}}
          STACK: ${{{{ inputs.stack }}}}
          SIZE: ${{{{ inputs.size }}}}
          PERSONAS: ${{{{ inputs.personas }}}}
        run: |
          args=(benchmark)
          add() {{
            if [[ ! "$2" =~ ^[A-Za-z0-9._/,-]+$ ]]; then
              echo "::error::$1 has characters a model or scenario name can't have: $2"
              exit 1
            fi
            args+=("$1" "$2")
          }}
          if [[ "$SCENARIO" == *"{repo_suffix}" ]]; then
            add --scenario "${{SCENARIO%{repo_suffix}}}"
          else
            add --assets "$SCENARIO"
            case "$SCENARIO" in
{prompt_cases}
            esac
          fi
          if [[ "$STACK" == *" + "* ]]; then
            IFS='+' read -r stt llm tts <<< "${{STACK// /}}"
            add --stt "$stt"
            add --llm "$llm"
            add --tts "$tts"
          else
            add --s2s "$STACK"
          fi
          case "$SIZE" in
            smoke) args+=(--smoke) ;;
            "sample "*) add --sample "${{SIZE#sample }}" ;;
            full) args+=(--full) ;;
            *) echo "::error::size must be smoke, sample N or full: $SIZE"; exit 1 ;;
          esac
          [ "$PERSONAS" != "default" ] && add --personas "$PERSONAS"
          echo "command=corral ${{args[*]}}" >> "$GITHUB_OUTPUT"
          echo "joined=$(IFS=';'; echo "${{args[*]}}")" >> "$GITHUB_OUTPUT"
      - uses: google-github-actions/auth@7c6bc770dae815cd3e89ee6cdf493a5fab2cc093 # v3.0.0
        with:
          workload_identity_provider: ${{{{ secrets.WIF_PROVIDER }}}}
          service_account: ${{{{ secrets.WIF_RUN_SA }}}}
      - uses: google-github-actions/setup-gcloud@aa5489c8933f4cc7a4f7d45035b3b1440c9c10db # v3.0.1
      - name: Run the corral job
        id: run
        env:
          COMMAND: ${{{{ steps.flags.outputs.command }}}}
          JOINED: ${{{{ steps.flags.outputs.joined }}}}
        run: |
          echo "$COMMAND"
          project=$(gcloud config get-value project 2>/dev/null)
          set +e
          gcloud run jobs execute corral --region us-east1 --wait --args="^;^${{JOINED}}" 2>&1 \\
            | tee execute.log
          status=${{PIPESTATUS[0]}}
          set -e
          execution=$(grep -oE 'Execution \\[[^]]+\\]' execute.log | head -1 | tr -d '[]' \\
            | cut -d' ' -f2 || true)
          image=""
          if [ -n "$execution" ]; then
            image=$(gcloud run jobs executions describe "$execution" --region us-east1 \\
              --format='value(spec.template.spec.containers[0].image)' || true)
          fi
          console=https://console.cloud.google.com/run/jobs/executions/details/us-east1
          logs="$console/${{execution}}/logs?project=${{project}}"
          {{
            echo '```'
            echo "$COMMAND"
            echo '```'
            echo
            if [ -n "$execution" ]; then
              echo "Cloud Run execution [\\`$execution\\`]($logs)"
            else
              echo "The job didn't start; see the step log above."
            fi
            echo
            echo "Image \\`${{image:-unknown}}\\`"
          }} >> "$GITHUB_STEP_SUMMARY"
          exit "$status"
"""


DAILY_TEMPLATE = """\
# Generated by scripts/gen_benchmark_workflow.py from the model catalogs and the daily rows in
# .github/benchmarks.json. Edit those and commit; a pre-commit hook rewrites this file and CI
# checks it's current.

name: Daily
run-name: Daily · ${{{{ github.event_name }}}}

on:
  schedule:
    - cron: "{cron}"
  workflow_dispatch:

permissions:
  contents: read
  id-token: write

jobs:
  benchmark:
    name: ${{{{ matrix.scenario }}}} · ${{{{ matrix.stack }}}}
    strategy:
      fail-fast: false
      max-parallel: {parallel}
      matrix:
        include:
{matrix}
    uses: ./.github/workflows/benchmark.yml
    with:
      scenario: ${{{{ matrix.scenario }}}}
      stack: ${{{{ matrix.stack }}}}
      size: ${{{{ matrix.size }}}}
      personas: ${{{{ matrix.personas }}}}
    secrets: inherit
"""


def main(argv: list[str]) -> int:
    stale = []
    for path, rendered in render().items():
        if "--check" in argv:
            current = path.read_text() if path.exists() else ""
            if current != rendered:
                stale.append(path)
            continue
        path.write_text(rendered)
        print(f"wrote {path}")
    for path in stale:
        print(f"{path} is stale; run: uv run python scripts/gen_benchmark_workflow.py")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
