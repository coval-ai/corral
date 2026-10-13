# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from corral.scenarios import load_suite, require_suite


def test_a_scenario_carries_its_coval_suite_or_is_for_talking_only() -> None:
    ultrabank = require_suite("ultrabank")
    assert ultrabank.test_set == "QjkAPGkb"
    assert ultrabank.workspace == "01KVGSYSYM8VWWE8M8Y5S14D5N"
    assert len(ultrabank.personas) == 1 and len(ultrabank.metrics) == 3
    assert load_suite("default") is None
    with pytest.raises(ValueError, match="'default' has no coval.json"):
        require_suite("default")
