# Copyright 2026 The Coval Benchmarks Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from corral.scenarios import load_test_set, require_test_set


def test_a_scenario_carries_its_coval_ids_or_is_for_talking_only() -> None:
    baseline = require_test_set("baseline")
    assert baseline.test_set_id == "QjkAPGkb"
    assert baseline.persona_id and len(baseline.metric_ids) == 3
    assert load_test_set("default") is None
    with pytest.raises(ValueError, match="'default' has no coval.json"):
        require_test_set("default")
