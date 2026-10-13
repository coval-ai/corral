# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One command. No arguments opens the menu; a mode and flags make it a script."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn

from corral import benchmark, serve, talk
from corral.agent import load_agent
from corral.config import Settings, get_settings
from corral.logging import configure_logging
from corral.models import MODALITIES, Modality, model_ids
from corral.scenarios import (
    BENCHMARK_SCENARIO,
    TALK_SCENARIO,
    load_test_set,
    require_test_set,
    scenario_names,
    scenario_summary,
)

Reader = Callable[[str], str]


@dataclass(frozen=True)
class Mode:
    name: str
    summary: str


TALK = Mode("talk", "to the agent")
BENCHMARK = Mode("benchmark", "with Coval")
SERVE = Mode("serve", "the image's command: serve on PORT with no tunnel and no run")
MODES = (TALK, BENCHMARK, SERVE)
MENU_MODES = (TALK, BENCHMARK)


def missing_for(mode: Mode) -> str | None:
    """What this machine still needs before the mode can run, or None."""
    if mode is TALK and importlib.util.find_spec("pyaudio") is None:
        return "brew install portaudio && uv sync --extra local"
    if mode is BENCHMARK and shutil.which("cloudflared") is None:
        return "brew install cloudflared"
    return None


def pick(options: list[str], prompt: str, read: Reader) -> int:
    """Print numbered options and return the index of the one chosen; Enter takes the first."""
    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")
    while True:
        try:
            answer = read(f"{prompt} [1-{len(options)}, default 1]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            raise SystemExit(130) from None
        if not answer:
            return 0
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print(f"pick a number from 1 to {len(options)}, or Enter for 1")


def choose_mode(read: Reader = input) -> Mode:
    width = max(len(mode.name) for mode in MENU_MODES)
    options = []
    for mode in MENU_MODES:
        line = f"{mode.name:<{width}}  {mode.summary}"
        needs = missing_for(mode)
        options.append(f"{line}  (needs: {needs})" if needs else line)
    return MENU_MODES[pick(options, "Mode", read)]


def scenarios_for(mode: Mode) -> list[str]:
    """Every scenario to talk to; only those with Coval ids to benchmark. Default first."""
    names = sorted(scenario_names(), key=lambda name: (name != TALK_SCENARIO, name))
    if mode is BENCHMARK:
        names = [name for name in names if load_test_set(name) is not None]
    return names


def choose_scenario(mode: Mode, read: Reader = input) -> str:
    names = scenarios_for(mode)
    width = max(len(name) for name in names)
    options = [f"{name:<{width}}  {scenario_summary(name)}" for name in names]
    return names[pick(options, "Scenario", read)]


def choose_model(modality: Modality, read: Reader = input) -> str:
    ids = model_ids(modality)
    return ids[pick(ids, modality.upper(), read)]


def has_coval_key(settings: Settings) -> bool:
    return settings.coval_api_key is not None and bool(settings.coval_api_key.get_secret_value())


def fail(parser: argparse.ArgumentParser, interactive: bool, message: str) -> NoReturn:
    """A plain line when someone is at the menu; usage plus the line when scripted."""
    if interactive:
        raise SystemExit(message)
    parser.error(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="corral", description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=[mode.name for mode in MODES],
        help="; ".join(f"{mode.name}: {mode.summary}" for mode in MODES),
    )
    for modality in MODALITIES:
        parser.add_argument(
            f"--{modality}",
            choices=model_ids(modality),
            default=os.environ.get(f"S2S_{modality.upper()}", model_ids(modality)[0]),
            help="default: %(default)s",
        )
    parser.add_argument(
        "--scenario",
        default=os.environ.get("S2S_SCENARIO"),
        help=f"default: {TALK_SCENARIO} to talk, {BENCHMARK_SCENARIO} otherwise",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    run = parser.add_argument_group("benchmark")
    run.add_argument("--smoke", action="store_true", help="one test case, to check the wiring")
    run.add_argument("--no-tunnel", action="store_true", help="probe localhost only")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    interactive = args.mode is None
    mode = choose_mode() if interactive else next(m for m in MODES if m.name == args.mode)
    needs = missing_for(mode)
    if needs and not (mode is BENCHMARK and args.no_tunnel):
        fail(parser, interactive, f"{mode.name} needs: {needs}")
    if mode is BENCHMARK and not has_coval_key(get_settings()):
        fail(parser, interactive, "a Coval run needs COVAL_API_KEY in the environment or .env")

    if args.scenario:
        scenario = args.scenario
    elif interactive and mode is not SERVE:
        scenario = choose_scenario(mode)
    else:
        scenario = TALK_SCENARIO if mode is TALK else BENCHMARK_SCENARIO
    request = None
    if mode is BENCHMARK:
        try:
            test_set = require_test_set(scenario)
        except ValueError as error:
            fail(parser, interactive, str(error))
        request = benchmark.RunRequest(test_set, sample=1 if args.smoke else 0)

    if interactive:
        agent = load_agent(scenario, *(choose_model(modality) for modality in MODALITIES))
    else:
        agent = load_agent(scenario, args.stt, args.llm, args.tts)
    configure_logging()
    with contextlib.suppress(KeyboardInterrupt):
        if mode is TALK:
            asyncio.run(talk.run(agent))
        elif mode is SERVE:
            serve.serve(agent, args.port)
        else:
            assert request is not None  # noqa: S101
            code = asyncio.run(benchmark.run(agent, args.port, not args.no_tunnel, request))
            raise SystemExit(code)


if __name__ == "__main__":
    main()
