"""Explicit opt-out for tests that execute the CUDA platform."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip-cuda", action="store_true", default=False,
        help="Skip CUDA execution tests on CPU-only runners.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-cuda"):
        skip = pytest.mark.skip(reason="CUDA execution disabled by --skip-cuda")
        for item in items:
            if item.get_closest_marker("cuda") is not None:
                item.add_marker(skip)
