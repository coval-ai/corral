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
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn

from corral import benchmark, serve, talk
from corral.agent import load_agent
from corral.config import Settings, get_settings
from corral.coval_api import SyncError, coval_problem
from corral.logging import configure_logging
from corral.models import (
    MODALITIES,
    Modality,
    Model,
    api_key,
    catalog,
    key_name,
    vendor_problem,
)
from corral.scenarios import (
    BENCHMARK_SCENARIO,
    TALK_SCENARIO,
    load_suite,
    require_suite,
    scenario_names,
    scenario_summary,
)

Reader = Callable[[str], str]
Checker = Callable[[str], str | None]


@dataclass(frozen=True)
class Mode:
    name: str
    summary: str


TALK = Mode("talk", "to the agent")
BENCHMARK = Mode("benchmark", "with Coval")
SERVE = Mode("serve", "the image's command: serve on PORT with no tunnel and no run")
MODES = (TALK, BENCHMARK, SERVE)
MENU_MODES = (TALK, BENCHMARK)
SERVE_PORT = 7873


def missing_for(mode: Mode, settings: Settings, tunnel: bool = True) -> str | None:
    """What this machine still needs before the mode can run, or None."""
    if mode is TALK and importlib.util.find_spec("miniaudio") is None:
        return "uv sync"
    if mode is BENCHMARK:
        if tunnel and shutil.which("cloudflared") is None:
            return "brew install cloudflared"
        if not has_coval_key(settings):
            return "COVAL_API_KEY in .env"
    return None


def pick(options: list[str], prompt: str, read: Reader, default: int = 0) -> int:
    """Print numbered options and return the index of the one chosen; Enter takes the default."""
    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")
    while True:
        try:
            answer = read(f"{prompt} [1-{len(options)}, default {default + 1}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            raise SystemExit(130) from None
        if not answer:
            return default
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print(f"pick a number from 1 to {len(options)}, or Enter for {default + 1}")


def choose_mode(read: Reader = input, settings: Settings | None = None) -> Mode:
    settings = settings or get_settings()
    width = max(len(mode.name) for mode in MENU_MODES)
    options = []
    for mode in MENU_MODES:
        line = f"{mode.name:<{width}}  {mode.summary}"
        needs = missing_for(mode, settings)
        options.append(f"{line}  (needs: {needs})" if needs else line)
    return MENU_MODES[pick(options, "Mode", read)]


def scenarios_for(mode: Mode) -> list[str]:
    """Every scenario to talk to; only those with Coval ids to benchmark. Default first."""
    names = sorted(scenario_names(), key=lambda name: (name != TALK_SCENARIO, name))
    if mode is BENCHMARK:
        names = [name for name in names if load_suite(name) is not None]
    return names


def known_scenario(name: str) -> str:
    """The name back, or why it isn't a scenario."""
    names = scenario_names()
    if name not in names:
        raise ValueError(f"no scenario {name!r}; have {', '.join(names)}")
    return name


def choose_scenario(mode: Mode, read: Reader = input) -> str:
    names = scenarios_for(mode)
    width = max(len(name) for name in names)
    options = [f"{name:<{width}}  {scenario_summary(name)}" for name in names]
    return names[pick(options, "Scenario", read)]


def checker(settings: Settings) -> Checker:
    """Each provider's problem, or None, asked of the vendor once and remembered."""
    answers: dict[str, str | None] = {}

    def check(provider: str) -> str | None:
        if provider not in answers:
            answers[provider] = vendor_problem(provider, settings)
        return answers[provider]

    return check


def offered_models(modality: Modality, settings: Settings) -> list[Model]:
    """The models whose vendor has a key in .env; the first is the default."""
    offered: list[Model] = getattr(catalog(), modality)
    keyed = [model for model in offered if api_key(model.provider, settings) is not None]
    if not keyed:
        names = sorted({key_name(model.provider) for model in offered})
        raise SystemExit(f"no {modality} model has a key; set one of {', '.join(names)} in .env")
    return keyed


def choose_model(
    modality: Modality,
    read: Reader = input,
    settings: Settings | None = None,
    check: Checker | None = None,
) -> str:
    """Pick a model whose vendor answers; Enter takes the first offered.

    A pick whose vendor fails its check says why and asks again.
    """
    settings = settings or get_settings()
    check = check or checker(settings)
    offered = offered_models(modality, settings)
    problems: dict[str, str] = {}
    while True:
        options = [
            f"{model.id}  ({problems[model.provider]})" if model.provider in problems else model.id
            for model in offered
        ]
        model = offered[pick(options, modality.upper(), read)]
        problem = check(model.provider)
        if problem is None:
            return model.id
        problems[model.provider] = problem
        print(problem)


def choose_port(mode: Mode, flag: int | None, env: str | None) -> int:
    """The flag, else PORT, else the image's port to serve and whatever is free otherwise."""
    if flag is not None:
        port = flag
    elif env:
        if not env.isdigit():
            raise ValueError(f"PORT is {env!r}, not a port number")
        port = int(env)
    else:
        return SERVE_PORT if mode is SERVE else 0
    if not 0 < port < 65536:
        raise ValueError(f"port {port} is out of range")
    return port


MODE_FLAGS = {"test_case": TALK, "smoke": BENCHMARK, "no_tunnel": BENCHMARK}


def misplaced_flag(mode: Mode, args: argparse.Namespace) -> str | None:
    """A flag that belongs to another mode, so it would be silently ignored."""
    for dest, owner in MODE_FLAGS.items():
        if getattr(args, dest) and mode is not owner:
            return f"--{dest.replace('_', '-')} is for {owner.name}"
    return None


def port_in_use(port: int) -> bool:
    """A benchmark binds loopback, and a port held there blocks serve's every-interface bind too."""
    if port == 0:
        return False
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return True
    return False


def has_coval_key(settings: Settings) -> bool:
    return settings.coval_api_key is not None and bool(settings.coval_api_key.get_secret_value())


def fail(parser: argparse.ArgumentParser, interactive: bool, message: str) -> NoReturn:
    """A plain line when someone is at the menu; usage plus the line when scripted."""
    if interactive:
        raise SystemExit(message)
    parser.error(message)


def build_parser(settings: Settings | None = None) -> argparse.ArgumentParser:
    settings = settings or get_settings()
    offered = {m: [model.id for model in offered_models(m, settings)] for m in MODALITIES}
    models = "\n".join(f"  {m}  {', '.join(ids)}" for m, ids in offered.items())
    parser = argparse.ArgumentParser(
        prog="corral",
        description=__doc__,
        epilog=f"models with a key in .env:\n{models}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=[mode.name for mode in MODES],
        metavar="mode",
        help="; ".join(f"{mode.name}: {mode.summary}" for mode in MODES),
    )
    for modality, ids in offered.items():
        parser.add_argument(
            f"--{modality}",
            choices=ids,
            metavar="MODEL",
            default=os.environ.get(f"S2S_{modality.upper()}", ids[0]),
            help="default: %(default)s",
        )
    parser.add_argument(
        "--scenario",
        default=os.environ.get("S2S_SCENARIO"),
        help=f"default: {TALK_SCENARIO} to talk, {BENCHMARK_SCENARIO} otherwise",
    )
    parser.add_argument(
        "--port",
        type=int,
        help=f"default: PORT, else {SERVE_PORT} to serve, any free port otherwise",
    )
    parser.add_argument_group("talk").add_argument(
        "--test-case", help="seed the scenario's tools for this test case"
    )
    run = parser.add_argument_group("benchmark")
    run.add_argument("--smoke", action="store_true", help="one test case, to check the wiring")
    run.add_argument("--no-tunnel", action="store_true", help="probe localhost only")
    return parser


def main() -> None:
    settings = get_settings()
    parser = build_parser(settings)
    args = parser.parse_args()
    interactive = args.mode is None
    if interactive:
        mode = choose_mode(settings=settings)
    else:
        mode = next(m for m in MODES if m.name == args.mode)
    needs = missing_for(mode, settings, tunnel=not args.no_tunnel)
    if needs:
        fail(parser, interactive, f"{mode.name} needs: {needs}")
    misplaced = misplaced_flag(mode, args)
    if misplaced:
        fail(parser, interactive, misplaced)
    try:
        port = choose_port(mode, args.port, os.environ.get("PORT"))
    except ValueError as error:
        fail(parser, interactive, str(error))
    if mode is not TALK and port_in_use(port):
        fail(parser, interactive, f"port {port} is in use; pass --port")
    if mode is BENCHMARK:
        # Coval answers for its key here, the way each vendor does for a model pick.
        problem = coval_problem(settings)
        if problem:
            fail(parser, interactive, problem)

    if args.scenario:
        try:
            scenario = known_scenario(args.scenario)
        except ValueError as error:
            fail(parser, interactive, str(error))
    elif interactive and mode is not SERVE:
        scenario = choose_scenario(mode)
    else:
        scenario = TALK_SCENARIO if mode is TALK else BENCHMARK_SCENARIO
    request = None
    if mode is BENCHMARK:
        try:
            suite = require_suite(scenario)
        except ValueError as error:
            fail(parser, interactive, str(error))
        workspace_id = suite.workspace or get_settings().corral_workspace_id
        request = benchmark.RunRequest(suite, workspace_id, sample=1 if args.smoke else 0)

    check = checker(settings)
    if interactive:
        picks = (choose_model(modality, settings=settings, check=check) for modality in MODALITIES)
        agent = load_agent(scenario, *picks)
    else:
        # argparse checks a flag against its choices, but not a default from the environment.
        try:
            agent = load_agent(scenario, args.stt, args.llm, args.tts)
        except ValueError as error:
            fail(parser, interactive, str(error))
    # The menu checked each pick as it was made; a scripted run checks them all here.
    for model in (agent.stt, agent.llm, agent.tts):
        problem = check(model.provider)
        if problem:
            fail(parser, interactive, f"{model.id}: {problem}")
    configure_logging()
    with contextlib.suppress(KeyboardInterrupt):
        if mode is TALK:
            try:
                asyncio.run(talk.run(agent, args.test_case))
            except ValueError as error:
                raise SystemExit(f"corral: {error}") from None
        elif mode is SERVE:
            serve.serve(agent, port)
        else:
            assert request is not None  # noqa: S101
            try:
                code = asyncio.run(benchmark.run(agent, port, not args.no_tunnel, request))
            except SyncError as error:
                raise SystemExit(f"corral: {error}") from None
            raise SystemExit(code)


if __name__ == "__main__":
    main()
