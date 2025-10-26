"""Shared pytest fixtures for the deepread project."""

import pathlib
from collections.abc import Iterator

import pytest


@pytest.fixture(scope="session")
def project_root() -> Iterator[pathlib.Path]:
    """Return repository root for convenience in tests."""
    yield pathlib.Path(__file__).resolve().parent.parent
