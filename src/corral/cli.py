# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

"""One command. No arguments opens the menu; a mode and flags make it a script."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, NoReturn

from corral import benchmark, talk
from corral.agent import Agent, load_asset_agent, load_cascade_agent, load_native_agent
from corral.assets import AssetRun, load_asset_run, resolve_assets
from corral.check_access import run as run_check_access
from corral.config import Settings, get_settings
from corral.coval_api import CovalClient, SyncError, coval_problem
from corral.logging import configure_logging
from corral.models import (
    CASCADE,
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


TALK = Mode("talk", "talk to the agent through your mic and speakers")
BENCHMARK = Mode("benchmark", "run the scenario's Coval suite against the agent")
CHECK_ACCESS = Mode("check-access", "probe every key in .env and every model it pays for")
MODES = (TALK, BENCHMARK, CHECK_ACCESS)


def missing_for(mode: Mode, settings: Settings) -> str | None:
    """What this machine still needs before the mode can run, or None."""
    if mode is TALK and importlib.util.find_spec("miniaudio") is None:
        return "uv sync"
    if mode is BENCHMARK and not has_coval_key(settings):
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
    width = max(len(mode.name) for mode in MODES)
    options = []
    for mode in MODES:
        line = f"{mode.name:<{width}}  {mode.summary}"
        needs = missing_for(mode, settings)
        options.append(f"{line}  (needs: {needs})" if needs else line)
    return MODES[pick(options, "Mode", read)]


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


def choose_test_case(scenario: str, read: Reader = input) -> str | None:
    """Which test case to talk to, when the scenario's suite names some; Enter takes the first."""
    suite = load_suite(scenario)
    cases = suite.keyed_test_cases() if suite else {}
    if not cases:
        return None
    width = max(len(name) for name in cases)
    options = [f"{name:<{width}}  {description}".rstrip() for name, description in cases.items()]
    return list(cases)[pick(options, "Test case", read)]


def checker(settings: Settings) -> Checker:
    """Each provider's problem, or None, asked of the vendor once and remembered."""
    answers: dict[str, str | None] = {}

    def check(provider: str) -> str | None:
        if provider not in answers:
            answers[provider] = vendor_problem(provider, settings)
        return answers[provider]

    return check


def offered_models(modality: Modality, settings: Settings) -> list[Model]:
    """The models whose vendor has a key in .env; the first is the default.

    A cascade needs one of each, so a modality with no key is fatal. A native
    model is optional, so no key just means the menu doesn't offer one.
    """
    offered: list[Model] = getattr(catalog(), modality)
    keyed = [model for model in offered if api_key(model.provider, settings) is not None]
    if not keyed and modality != "s2s":
        names = sorted({key_name(model.provider) for model in offered})
        raise SystemExit(f"no {modality} model has a key; set one of {', '.join(names)} in .env")
    return keyed


def choose_agent(
    scenario: str,
    read: Reader = input,
    settings: Settings | None = None,
    check: Checker | None = None,
) -> Agent:
    """Cascade or native when there's a native model to offer, then a model per modality."""
    settings = settings or get_settings()
    check = check or checker(settings)
    native = False
    if offered_models("s2s", settings):
        options = ["cascade  an STT, an LLM and a TTS", "native   one speech-to-speech model"]
        native = pick(options, "Stack", read) == 1
    if native:
        return load_native_agent(scenario, choose_model("s2s", read, settings, check))
    picks = (choose_model(modality, read, settings, check) for modality in CASCADE)
    return load_cascade_agent(scenario, *picks)


def flagged_agent(scenario: str, args: argparse.Namespace) -> Agent:
    """The stack the flags name: one native model, else the cascade."""
    if args.s2s:
        return load_native_agent(scenario, args.s2s)
    return load_cascade_agent(scenario, args.stt, args.llm, args.tts)


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


def needs_assets(args: argparse.Namespace) -> str | None:
    """A selector that only means something with --assets, so it would be silently ignored."""
    for dest in ("personas", "metrics", "prompt_from"):
        if getattr(args, dest) and not args.assets:
            return f"--{dest.replace('_', '-')} needs --assets"
    return None


def has_coval_key(settings: Settings) -> bool:
    return settings.coval_api_key is not None and bool(settings.coval_api_key.get_secret_value())


def fail(parser: argparse.ArgumentParser, interactive: bool, message: str) -> NoReturn:
    """A plain line when someone is at the menu; usage plus the line when scripted."""
    if interactive:
        raise SystemExit(message)
    parser.error(message)


def build_parser(settings: Settings | None = None) -> argparse.ArgumentParser:
    """``corral`` alone is the menu. A mode makes it a script, and each mode has its own help."""
    settings = settings or get_settings()
    offered = {m: [model.id for model in offered_models(m, settings)] for m in MODALITIES}
    models = "\n".join(f"  {m}  {', '.join(ids)}" for m, ids in offered.items() if ids)
    parser = argparse.ArgumentParser(
        prog="corral",
        description="Wrangle a voice agent: talk to it, benchmark it on Coval, or check its keys."
        " With no mode, a menu asks.",
    )
    modes = parser.add_subparsers(dest="mode", metavar="mode", title="modes")
    defaults: dict[str, Any] = {
        "mode": None,
        "scenario": os.environ.get("S2S_SCENARIO"),
        "test_case": None,
        "smoke": False,
        "assets": os.environ.get("S2S_ASSETS"),
        "personas": [],
        "metrics": [],
        "prompt_from": None,
    }
    agent = argparse.ArgumentParser(add_help=False)
    for modality, ids in offered.items():
        native = modality == "s2s"
        default = os.environ.get(f"S2S_{modality.upper()}", None if native or not ids else ids[0])
        defaults[modality] = default
        if not ids:
            continue
        agent.add_argument(
            f"--{modality}",
            choices=ids,
            metavar="MODEL",
            default=default,
            help="one native model in place of the three" if native else "default: %(default)s",
        )
    agent.add_argument(
        "--scenario",
        default=defaults["scenario"],
        help=f"default: {TALK_SCENARIO} to talk, {BENCHMARK_SCENARIO} to benchmark",
    )
    parser.set_defaults(**defaults)
    epilog = f"models with a key in .env:\n{models}"
    talk = modes.add_parser(
        TALK.name,
        help=TALK.summary,
        description=TALK.summary.capitalize() + ".",
        parents=[agent],
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    talk.add_argument("--test-case", help="seed the scenario's tools for this test case")
    benchmark = modes.add_parser(
        BENCHMARK.name,
        help=BENCHMARK.summary,
        description=BENCHMARK.summary.capitalize() + ".",
        parents=[agent],
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    benchmark.add_argument(
        "--smoke", action="store_true", help="one test case, to check the wiring"
    )
    benchmark.add_argument(
        "--assets",
        default=defaults["assets"],
        metavar="SCENARIO.VERSION",
        help="run a scenario from the manifest at CORRAL_ASSETS_URI instead of --scenario",
    )
    benchmark.add_argument(
        "--personas", type=names, default=[], help="with --assets: conditions to run; default all"
    )
    benchmark.add_argument(
        "--metrics", type=names, default=[], help="with --assets: metrics to score; default all"
    )
    benchmark.add_argument(
        "--prompt-from",
        metavar="MODEL",
        help="with --assets: the agent whose prompt to use when the model isn't one of them",
    )
    modes.add_parser(
        CHECK_ACCESS.name,
        help=CHECK_ACCESS.summary,
        description=CHECK_ACCESS.summary.capitalize()
        + ". One line per vendor and model; exits 1 when anything with a key fails.",
    )
    return parser


def names(value: str) -> list[str]:
    """A comma-separated flag as its names."""
    return [name.strip() for name in value.split(",") if name.strip()]


def asset_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> AssetRun:
    """The suite and prompt ``--assets`` names; Coval is asked for the agent's prompt."""
    settings = get_settings()
    api_key = settings.coval_api_key.get_secret_value() if settings.coval_api_key else ""
    try:
        with CovalClient(api_key, settings.coval_api_base) as coval:
            return load_asset_run(
                args.assets,
                personas=args.personas,
                metrics=args.metrics,
                model_id=args.s2s or args.llm,
                prompt_from=args.prompt_from,
                settings=settings,
                client=coval,
            )
    except (SyncError, ValueError) as error:
        fail(parser, False, str(error))


def main() -> None:
    settings = get_settings()
    parser = build_parser(settings)
    args = parser.parse_args()
    interactive = args.mode is None
    if interactive:
        mode = choose_mode(settings=settings)
    else:
        mode = next(m for m in MODES if m.name == args.mode)
    needs = missing_for(mode, settings)
    if needs:
        fail(parser, interactive, f"{mode.name} needs: {needs}")
    if mode is CHECK_ACCESS:
        raise SystemExit(asyncio.run(run_check_access(settings)))
    misplaced = needs_assets(args)
    if misplaced:
        fail(parser, interactive, misplaced)
    if mode is BENCHMARK:
        # Coval answers for its key here, the way each vendor does for a model pick.
        problem = coval_problem(settings)
        if problem:
            fail(parser, interactive, problem)

    assets = None
    if args.assets:
        if args.scenario:
            fail(parser, interactive, "--assets names the scenario; drop --scenario")
        if interactive:
            fail(parser, interactive, "--assets runs from flags: corral benchmark --assets ...")
        assets = asset_run(parser, args)
        scenario = args.assets
    elif args.scenario:
        try:
            scenario = known_scenario(args.scenario)
        except ValueError as error:
            fail(parser, interactive, str(error))
    elif interactive:
        scenario = choose_scenario(mode)
    else:
        scenario = TALK_SCENARIO if mode is TALK else BENCHMARK_SCENARIO
    request = None
    if assets:
        request = benchmark.run_request(
            assets.suite, assets.suite.workspace, smoke=args.smoke, today=datetime.now(UTC).date()
        )
        request = replace(request, workspace_source=assets.source)
    elif mode is BENCHMARK:
        try:
            named = require_suite(scenario)
            suite = resolve_assets(named, get_settings())
        except ValueError as error:
            fail(parser, interactive, str(error))
        workspace_id = suite.workspace or get_settings().corral_workspace_id
        request = benchmark.run_request(
            suite, workspace_id, smoke=args.smoke, today=datetime.now(UTC).date()
        )
        if named.assets:
            source = f"{get_settings().corral_assets_uri} ({named.assets})"
            request = replace(request, workspace_source=source)

    test_case = args.test_case
    if interactive and mode is TALK:
        test_case = choose_test_case(scenario)
    check = checker(settings)
    if assets:
        try:
            agent = load_asset_agent(
                args.assets, assets.system_prompt, args.s2s, args.stt, args.llm, args.tts
            )
        except ValueError as error:
            fail(parser, interactive, str(error))
    elif interactive:
        agent = choose_agent(scenario, settings=settings, check=check)
    else:
        # argparse checks a flag against its choices, but not a default from the environment.
        try:
            agent = flagged_agent(scenario, args)
        except ValueError as error:
            fail(parser, interactive, str(error))
    # The menu checked each pick as it was made; a scripted run checks them all here.
    for model in agent.stack.rows:
        problem = check(model.provider)
        if problem:
            fail(parser, interactive, f"{model.id}: {problem}")
    configure_logging()
    with contextlib.suppress(KeyboardInterrupt):
        if mode is TALK:
            try:
                asyncio.run(talk.run(agent, test_case))
            except ValueError as error:
                raise SystemExit(f"corral: {error}") from None
        else:
            assert request is not None  # noqa: S101
            try:
                code = asyncio.run(benchmark.run(agent, request))
            except (SyncError, ValueError) as error:
                raise SystemExit(f"corral: {error}") from None
            raise SystemExit(code)


if __name__ == "__main__":
    main()
