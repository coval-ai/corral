# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One command. With no arguments it asks for a mode and a stack; with them it is a script."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
from collections.abc import Callable
from dataclasses import dataclass

from harness import app, driver, serve
from harness.config import Settings, get_settings
from harness.coval_agent import METRIC_SETTINGS, metric_ids
from harness.logging import configure_logging
from harness.stack import DEFAULT_SCENARIO, DEFAULT_STACK, LoadedStack, load_stack, stack_slugs

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
    options = [f"{mode.name:<{width}}  {mode.summary}" for mode in MENU_MODES]
    return MENU_MODES[pick(options, "Mode", read)]


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


def missing_for_benchmark(
    settings: Settings, persona_id: str | None, test_set_id: str | None
) -> list[str]:
    """The env vars a Coval run still needs, as the user would spell them."""
    missing = []
    if settings.coval_api_key is None or not settings.coval_api_key.get_secret_value():
        missing.append("COVAL_API_KEY")
    if not persona_id:
        missing.append("COVAL_S2S_BANK_PERSONA_ID")
    if not test_set_id:
        missing.append("COVAL_S2S_BANK_TEST_SET_ID")
    missing += [name.upper() for name in METRIC_SETTINGS if not (getattr(settings, name) or "")]
    return missing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="harness", description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=[mode.name for mode in MODES],
        help="; ".join(f"{mode.name}: {mode.summary}" for mode in MODES),
    )
    parser.add_argument("--stack", default=os.environ.get("S2S_STACK", DEFAULT_STACK))
    parser.add_argument("--scenario", default=os.environ.get("S2S_SCENARIO", DEFAULT_SCENARIO))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    run = parser.add_argument_group("benchmark")
    run.add_argument("--sample", type=int, default=0, help="test cases to run; 0 is all")
    run.add_argument("--persona-id", default=None, help="default: COVAL_S2S_BANK_PERSONA_ID")
    run.add_argument("--test-set-id", default=None, help="default: COVAL_S2S_BANK_TEST_SET_ID")
    run.add_argument("--no-tunnel", action="store_true", help="probe localhost only")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    interactive = args.mode is None
    mode = choose_mode() if interactive else next(m for m in MODES if m.name == args.mode)

    request = None
    if mode is BENCHMARK:
        settings = get_settings()
        persona_id = args.persona_id or settings.coval_s2s_bank_persona_id
        test_set_id = args.test_set_id or settings.coval_s2s_bank_test_set_id
        missing = missing_for_benchmark(settings, persona_id, test_set_id)
        if missing or not persona_id or not test_set_id:
            message = f"a Coval run needs {', '.join(missing)} in the environment or .env"
            if interactive:
                raise SystemExit(message)
            parser.error(message)
        request = driver.RunRequest(persona_id, test_set_id, metric_ids(settings), args.sample)

    loaded = choose_stack(args.scenario) if interactive else load_stack(args.scenario, args.stack)
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
