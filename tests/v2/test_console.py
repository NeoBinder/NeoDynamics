"""Public-interface tests for neomd.console + the driver/CLI printing
behavior it enables.

The printing contract:

* every run spelling (md_run, compile().run(), direct drive(), the CLI)
  prints by default — drive() calls ensure_console_logging(), which
  attaches ONE InlineProgressHandler to the "neomd" package logger and
  lifts a NOTSET level to INFO (explicit levels are respected, which is
  how ``--silent`` holds);
* drive() brackets the run with a start banner (start time, method, every
  input file, the output path) and an end line (end time + elapsed);
* run_md's periodic progress records carry ``inline=True`` and render as
  same-line replacements through InlineProgressHandler; the final 100%
  line stays in the scrollback;
* ``neomd run --silent`` prints nothing (no banner, no progress, no
  summary) and restores the logging level afterwards.

Discipline §8 #5: assertions observe logging records (caplog), stream
content (StringIO / capsys), and exit codes — no internals are probed.
"""

from __future__ import annotations

import io
import logging

import pytest

from neomd.console import InlineProgressHandler, ensure_console_logging
from neomd.driver import drive, run_md
from neomd.kernel import KernelSpec
from neomd.kernel._bootstrap import ensure_adapters
from neomd.kernel.fake import FakeKernel
from neomd.plan import Plan

ensure_adapters()

PACKAGE_LOGGER = "neomd"
DRIVER_LOGGER = "neomd.driver"


@pytest.fixture(autouse=True)
def clean_console_state():
    """Undo ensure_console_logging()'s process-global attach so each test
    starts from the library default (no handler, NOTSET level)."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    logger.handlers = [h for h in logger.handlers
                       if not isinstance(h, InlineProgressHandler)]
    yield
    logger.handlers = saved_handlers
    logger.setLevel(saved_level)


def fake_plan(**overrides) -> Plan:
    config = {
        "method": "eq",
        "steps": 100,
        "temperature": 298,
        "seed": 42,
        "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": "/tmp/neomd-console-test",
                   "state_interval": 0, "trajectory_interval": 0,
                   "checkpoint_interval": 0},
    }
    config.update(overrides)
    return Plan.from_dict(config)


def fake_kernel(seed: int = 1) -> FakeKernel:
    return FakeKernel(KernelSpec(kind="fake", seed=seed, temperature=298.0))


# ---------------------------------------------------------------------------
# ensure_console_logging
# ---------------------------------------------------------------------------


def test_ensure_attaches_one_handler_and_lifts_notset_to_info():
    logger = ensure_console_logging()
    assert logger.name == PACKAGE_LOGGER
    handlers = [h for h in logger.handlers
                if isinstance(h, InlineProgressHandler)]
    assert len(handlers) == 1
    assert logger.level == logging.INFO

    ensure_console_logging()  # idempotent: still exactly one handler
    handlers = [h for h in logger.handlers
                if isinstance(h, InlineProgressHandler)]
    assert len(handlers) == 1


def test_ensure_respects_an_explicit_level():
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(logging.CRITICAL + 1)  # the --silent pin
    ensure_console_logging()
    assert logger.level == logging.CRITICAL + 1


# ---------------------------------------------------------------------------
# InlineProgressHandler — same-line replacement rendering
# ---------------------------------------------------------------------------


def _record(msg: str, inline: bool) -> logging.LogRecord:
    record = logging.LogRecord(DRIVER_LOGGER, logging.INFO, __file__, 0,
                               msg, None, None)
    record.inline = inline
    return record


def test_inline_records_replace_the_same_line():
    stream = io.StringIO()
    handler = InlineProgressHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))

    handler.emit(_record("run start: ...", inline=False))
    handler.emit(_record("progress 50%", inline=True))
    handler.emit(_record("progress 100%", inline=True))
    handler.emit(_record("run end: ...", inline=False))

    text = stream.getvalue()
    lines = text.split("\n")
    # the two inline writes share one physical line via carriage returns,
    # the second padded to erase the first
    assert lines[1].count("\r") == 2
    assert "progress 50%" in lines[1] and "progress 100%" in lines[1]
    # an ordinary record closes the inline line first: 3 records -> 3 lines
    assert [line.lstrip("\r")[:9] for line in lines[:3]] == [
        "run start", "progress ", "run end: "]


# ---------------------------------------------------------------------------
# drive() — banner, end line, default printing
# ---------------------------------------------------------------------------


def test_drive_logs_banner_and_end(caplog):
    plan = fake_plan(steps=20)
    kernel = fake_kernel()
    with caplog.at_level(logging.INFO, logger=DRIVER_LOGGER):
        drive(plan, kernel_factory=lambda spec: kernel)

    lines = [r.getMessage() for r in caplog.records
             if r.name == DRIVER_LOGGER]
    assert lines[0].startswith("run start: ")
    assert "method=eq" in lines[0]
    assert "input complex: unused.pdb" in lines
    assert "input system: unused.xml" in lines
    assert "output: /tmp/neomd-console-test" in lines
    assert lines[-1].startswith("run end: ")
    assert "elapsed" in lines[-1]


def test_drive_attaches_the_default_console_handler():
    logger = logging.getLogger(PACKAGE_LOGGER)
    assert not any(isinstance(h, InlineProgressHandler)
                   for h in logger.handlers)  # fixture-clean start
    kernel = fake_kernel()
    drive(fake_plan(steps=10), kernel_factory=lambda spec: kernel)
    assert any(isinstance(h, InlineProgressHandler)
               for h in logger.handlers)
    assert logger.level == logging.INFO


def test_drive_prints_banner_and_progress_to_stderr_by_default(capsys):
    kernel = fake_kernel()
    drive(fake_plan(steps=20), kernel_factory=lambda spec: kernel)
    err = capsys.readouterr().err
    assert "run start: " in err
    assert "input complex: unused.pdb" in err
    assert "output: /tmp/neomd-console-test" in err
    assert "run end: " in err
    assert "预计结束" in err  # the v1 ETA line


# ---------------------------------------------------------------------------
# run_md — inline progress records
# ---------------------------------------------------------------------------


def test_progress_records_are_inline_until_the_final_line(caplog):
    with caplog.at_level(logging.INFO, logger=DRIVER_LOGGER):
        run_md(fake_kernel(seed=3), fake_plan(steps=100), log_interval=50)
    records = [r for r in caplog.records if r.name == DRIVER_LOGGER]
    assert records[0].getMessage() == "current steps:0 remaining steps:100"
    assert getattr(records[0], "inline", False) is False

    progress = [r for r in records if "已完成" in r.getMessage()]
    assert len(progress) == 2
    assert getattr(progress[0], "inline") is True       # 50%: same-line
    assert getattr(progress[1], "inline", False) is False  # 100%: stays
    assert "100.00%" in progress[1].getMessage()
