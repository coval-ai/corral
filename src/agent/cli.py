# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One command. With no arguments it asks for a mode and a stack; with them it is a script."""

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

from agent import app, driver, serve
from agent.config import Settings, get_settings
from agent.logging import configure_logging
from agent.stack import (
    DEFAULT_SCENARIO,
    DEFAULT_STACK,
    TALK_SCENARIO,
    LoadedStack,
    load_stack,
    stack_slugs,
)
from agent.testset import DEFAULT_TEST_SET, TestSet, load_test_set, load_test_sets

Reader = Callable[[str], str]


@dataclass(frozen=True)
class Mode:
    name: str
    summary: str


TALK = Mode("talk", "microphone and speakers on this machine")
BENCHMARK = Mode("benchmark", "serve behind a tunnel, run it on Coval, exit")
SERVE = Mode("serve", "the image's command: serve on PORT with no tunnel and no run")
MODES = (TALK, BENCHMARK, SERVE)
MENU_MODES = (TALK, BENCHMARK)  # serve is for the image, not for someone at a terminal


def missing_for(mode: Mode) -> str | None:
    """What this machine still needs before the mode can run, or None."""
    if mode is TALK and importlib.util.find_spec("pyaudio") is None:
        return "brew install portaudio && uv sync --extra local"
    if mode is BENCHMARK and shutil.which("cloudflared") is None:
        return "brew install cloudflared"
    return None


def pick(options: list[str], prompt: str, read: Reader) -> int:
    """Print numbered options and return the index of the one chosen."""
    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")
    while True:
        answer = read(f"{prompt} [1-{len(options)}]: ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print(f"pick a number from 1 to {len(options)}")


def choose_mode(read: Reader = input) -> Mode:
    width = max(len(mode.name) for mode in MENU_MODES)
    options = []
    for mode in MENU_MODES:
        line = f"{mode.name:<{width}}  {mode.summary}"
        needs = missing_for(mode)
        options.append(f"{line}  (needs: {needs})" if needs else line)
    return MENU_MODES[pick(options, "Mode", read)]


def choose_test_set(scenario: str, read: Reader = input) -> TestSet:
    sets = load_test_sets(scenario)
    names = sorted(sets)
    width = max(len(name) for name in names)
    options = [f"{name:<{width}}  {sets[name].summary}" for name in names]
    return sets[names[pick(options, "Test set", read)]]


def choose_stack(scenario: str, read: Reader = input) -> LoadedStack:
    """Print every stack's STT, LLM and TTS and take a number."""
    stacks = [load_stack(scenario, slug) for slug in stack_slugs()]
    width = max(len(loaded.slug) for loaded in stacks)
    options = [
        f"{loaded.slug:<{width}}  "
        f"stt {loaded.stack.stt.provider}/{loaded.stack.stt.model}  "
        f"llm {loaded.stack.llm.provider}/{loaded.stack.llm.model}  "
        f"tts {loaded.stack.tts.provider}/{loaded.stack.tts.model}"
        for loaded in stacks
    ]
    return stacks[pick(options, "Stack", read)]


def has_coval_key(settings: Settings) -> bool:
    return settings.coval_api_key is not None and bool(settings.coval_api_key.get_secret_value())


def fail(parser: argparse.ArgumentParser, interactive: bool, message: str) -> NoReturn:
    """A plain line when someone is at the menu; usage plus the line when scripted."""
    if interactive:
        raise SystemExit(message)
    parser.error(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent", description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=[mode.name for mode in MODES],
        help="; ".join(f"{mode.name}: {mode.summary}" for mode in MODES),
    )
    parser.add_argument("--stack", default=os.environ.get("S2S_STACK", DEFAULT_STACK))
    parser.add_argument(
        "--scenario",
        default=os.environ.get("S2S_SCENARIO"),
        help=f"default: {TALK_SCENARIO} to talk, {DEFAULT_SCENARIO} otherwise",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    run = parser.add_argument_group("benchmark")
    run.add_argument(
        "--test-set",
        default=os.environ.get("S2S_TEST_SET", DEFAULT_TEST_SET),
        help="a name from the scenario's test-sets.json",
    )
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
    scenario = args.scenario or (TALK_SCENARIO if mode is TALK else DEFAULT_SCENARIO)

    request = None
    if mode is BENCHMARK:
        if not has_coval_key(get_settings()):
            fail(parser, interactive, "a Coval run needs COVAL_API_KEY in the environment or .env")
        test_set = (
            choose_test_set(scenario) if interactive else load_test_set(scenario, args.test_set)
        )
        request = driver.RunRequest(test_set, sample=1 if args.smoke else 0)

    loaded = choose_stack(scenario) if interactive else load_stack(scenario, args.stack)
    configure_logging()
    with contextlib.suppress(KeyboardInterrupt):
        if mode is TALK:
            asyncio.run(app.run(loaded))
        elif mode is SERVE:
            serve.serve(loaded, args.port)
        else:
            assert request is not None  # noqa: S101
            code = asyncio.run(driver.run(loaded, args.port, not args.no_tunnel, request))
            raise SystemExit(code)


if __name__ == "__main__":
    main()
