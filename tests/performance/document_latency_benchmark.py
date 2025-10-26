"""
Pipeline latency benchmark harness.

This test serves as a placeholder until the full benchmarking scripts are
implemented. It anchors the command structure referenced in the quickstart and
ensures pytest discovers the performance suite.
"""

from __future__ import annotations

import pytest


@pytest.mark.performance
def test_latency_benchmark_placeholder() -> None:
    pytest.skip("Latency benchmark harness not yet implemented.")
