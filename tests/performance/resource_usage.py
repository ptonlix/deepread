"""
Resource usage sampling harness.

Tracks memory and CPU envelopes for pipeline execution. Currently a stub so
the performance suite has a consistent entry point referenced in docs.
"""

from __future__ import annotations

import pytest


@pytest.mark.performance
def test_resource_usage_placeholder() -> None:
    pytest.skip("Resource usage measurements will be added in later phases.")
