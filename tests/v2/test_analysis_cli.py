"""Public-interface tests for ``neomd analysis`` — the CLI surface of the
analysis subpackage (issue #16, W1-a).

Everything crosses public seams only: ``neomd.cli.main(argv)`` over run
directories produced by real (fake-kernel) metadynamics drives, plus one
synthetic smd.tsv written in the documented format for the direct-tape
path.  Assertions observe exit codes, stdout/stderr and the files the
commands write; no CLI internals are probed.
"""

from __future__ import annotations

import json
import math
import os

# Determinism pin — before any kernel can exist in this process.
os.environ.setdefault("OPENMM_CPU_THREADS", "1")

import pathlib

import numpy as np
import pytest

from neomd.cli import main
from neomd.driver import drive
from neomd.kernel.fake import FakeKernel
from neomd.plan import Plan
from neomd.sinks import LocalDirSink

GRID_BINS = 40


def meta_run(directory, seed=2026, steps=200) -> pathlib.Path:
    """One small fake-kernel metadynamics run into ``directory`` (the
    test_metadynamics recipe: FakeKernel, 10 hills, all artifacts)."""
    config = {
        "method": "metadynamics", "steps": steps, "temperature": 298,
        "seed": seed, "integrator": {"dt": 0.002, "friction_coeff": 1.0},
        "input_files": {"complex": "unused.pdb", "system": "unused.xml"},
        "output": {"output_dir": str(directory), "state_interval": 0,
                   "trajectory_interval": 0, "checkpoint_interval": 0},
        "colvars": {"dist": {
            "type": "distance", "grp1_idx": "0", "grp2_idx": "1",
            "min_cv_nm": 0.5, "max_cv_nm": 3.5, "biasWidth_nm": 0.3,
            "bins": GRID_BINS}},
        "meta_set": {"biasFactor": 5.0, "height": 1.2, "frequency": 20},
    }
    outcome = drive(Plan.from_dict(config),
                    kernel_factory=lambda spec: FakeKernel(spec),
                    sink=LocalDirSink(directory))
    assert outcome.phases_run == ["metadynamics"]
    assert outcome.results[0].n_hills == steps // 20
    return pathlib.Path(directory)


def parse_tsv(stdout: str):
    """(header list without '#', row dicts) of a comment-headed tsv payload.

    The LAST comment line before the data is the column header (payloads
    may carry extra '#' summary lines above it).
    """
    header = None
    rows = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header = stripped.lstrip("#").strip().split("\t")
            continue
        rows.append(dict(zip(header, stripped.split("\t"))))
    return header, rows


# ---------------------------------------------------------------------------
# fes
# ---------------------------------------------------------------------------


def test_fes_end_to_end_on_a_fake_run(tmp_path, capsys):
    run = meta_run(tmp_path / "run")
    assert main(["analysis", "fes", str(run)]) == 0
    header, rows = parse_tsv(capsys.readouterr().out)
    assert header == ["dist [nm]", "fes [kJ/mol]"]
    assert len(rows) == GRID_BINS
    xs = [float(row["dist [nm]"]) for row in rows]
    assert xs[0] == pytest.approx(0.5) and xs[-1] == pytest.approx(3.5)
    values = np.array([float(row["fes [kJ/mol]"]) for row in rows])
    assert np.isfinite(values).all()
    assert (values <= 0.0).all()  # -(T+dT)/dT * nonnegative bias

    # --upto-step cuts the ledger (a mid-run surface is shallower)
    assert main(["analysis", "fes", str(run), "--upto-step", "100"]) == 0
    _, early_rows = parse_tsv(capsys.readouterr().out)
    early = np.array([float(row["fes [kJ/mol]"]) for row in early_rows])
    assert early.max() > values.max()  # fewer hills -> less bias filled in

    # --bins evaluates a custom-resolution grid
    assert main(["analysis", "fes", str(run), "--bins", "17"]) == 0
    _, fine_rows = parse_tsv(capsys.readouterr().out)
    assert len(fine_rows) == 17


def test_fes_matches_the_producer_fes_tsv(tmp_path, capsys):
    """The CLI's stdout payload equals the fes.tsv the run itself wrote —
    the strongest format + math parity pin at the CLI tier."""
    run = meta_run(tmp_path / "run")
    assert main(["analysis", "fes", str(run)]) == 0
    stdout = capsys.readouterr().out
    producer = (run / "fes.tsv").read_text()
    assert stdout == producer


def test_fes_out_file_gets_payload_and_summary(tmp_path, capsys):
    run = meta_run(tmp_path / "run")
    out = tmp_path / "fes_out.tsv"
    assert main(["analysis", "fes", str(run), "--out", str(out)]) == 0
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert lines == [f"analysis complete: tool=fes hills=10 cvs=dist "
                     f"out={out}"]
    assert out.read_text() == (run / "fes.tsv").read_text()


def test_fes_overrides_and_errors(tmp_path, capsys):
    run = meta_run(tmp_path / "run")
    # a different gamma rescales the whole surface
    assert main(["analysis", "fes", str(run), "--bias-factor", "9"]) == 0
    _, rows = parse_tsv(capsys.readouterr().out)
    rescaled = np.array([float(row["fes [kJ/mol]"]) for row in rows])
    assert main(["analysis", "fes", str(run)]) == 0
    _, rows = parse_tsv(capsys.readouterr().out)
    plain = np.array([float(row["fes [kJ/mol]"]) for row in rows])
    ratio = rescaled / plain
    assert np.allclose(ratio, -(9 / 8) / -(5 / 4), rtol=1e-12)

    assert main(["analysis", "fes", str(run), "--bias-factor", "1"]) == 2
    stderr = capsys.readouterr().err
    assert "bias-factor must be > 1.0" in stderr
    assert "Traceback" not in stderr


def test_missing_run_dir_is_a_clean_error(tmp_path, capsys):
    assert main(["analysis", "fes", str(tmp_path / "nope")]) == 2
    stderr = capsys.readouterr().err
    assert "analysis error" in stderr
    assert "Traceback" not in stderr

    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["analysis", "convergence", str(empty)]) == 2
    assert "no manifest.json" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------


def test_convergence_end_to_end(tmp_path, capsys):
    run = meta_run(tmp_path / "run")
    assert main(["analysis", "convergence", str(run), "--blocks", "2"]) == 0
    header, rows = parse_tsv(capsys.readouterr().out)
    assert header is not None and len(header) == 6
    assert len(rows) == 2
    first, last = rows
    assert math.isnan(float(first["max_abs_dF_prev [kJ/mol]"]))
    assert float(last["max_abs_dF_final [kJ/mol]"]) == 0.0
    assert float(last["mean_abs_dF_final [kJ/mol]"]) == 0.0
    assert int(first["n_hills"]) == 5 and int(last["n_hills"]) == 10
    assert int(last["last_step"]) == 200
    prev = abs(float(last["max_abs_dF_prev [kJ/mol]"]))
    assert prev > 0.0  # the surface moved between halves


# ---------------------------------------------------------------------------
# block-average
# ---------------------------------------------------------------------------


def test_block_average_on_a_run_column(tmp_path, capsys):
    run = meta_run(tmp_path / "run")
    assert main(["analysis", "block-average", str(run),
                 "--column", "dist"]) == 0
    captured = capsys.readouterr().out
    comments = [line for line in captured.splitlines()
                if line.startswith("#")]
    assert any("column=dist" in line for line in comments)
    assert any("statistical error" in line for line in comments)
    header, rows = parse_tsv(captured)
    assert header == ["block_size", "n_blocks", "sem"]
    assert rows and int(rows[0]["block_size"]) == 1
    assert all(float(row["sem"]) >= 0.0 for row in rows)

    # did-you-mean on a misspelled column
    assert main(["analysis", "block-average", str(run),
                 "--column", "distance"]) == 2
    assert "did you mean" in capsys.readouterr().err


def test_block_average_on_a_direct_smd_tape(tmp_path, capsys):
    """The documented smd.tsv format, read directly as a tape file."""
    tape = tmp_path / "smd.tsv"
    tape.write_text(
        "# step\tpull\tpull__restr_k\tpull__energy\n"
        "3000\t0.51\t0.0\t0.0\n"
        "6000\t0.62\t83.33\t0.2\n"
        "9000\t0.58\t83.33\t0.5\n"
        "12000\t0.71\t100.0\t1.2\n"
    )
    assert main(["analysis", "block-average", str(tape),
                 "--column", "pull__energy", "--min-blocks", "2"]) == 0
    captured = capsys.readouterr().out
    assert any("column=pull__energy" in line
               for line in captured.splitlines() if line.startswith("#"))
    header, rows = parse_tsv(captured)
    assert header == ["block_size", "n_blocks", "sem"]
    assert len(rows) == 2  # b=1 (4 blocks), b=2 (2 blocks)


# ---------------------------------------------------------------------------
# reweight
# ---------------------------------------------------------------------------


def test_reweight_end_to_end(tmp_path, capsys):
    run = meta_run(tmp_path / "run")
    assert main(["analysis", "reweight", str(run),
                 "--observable", "dist"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["observable"] == "dist"
    assert math.isfinite(payload["mean"])
    assert 0.0 < payload["ess"] <= 10  # at most one per biased sample
    assert payload["n_used"] == 10
    assert payload["runs"] == 1

    # --out files the json, summary line lands on stdout
    out = tmp_path / "rw.json"
    assert main(["analysis", "reweight", str(run), "--observable", "dist",
                 "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert json.loads(out.read_text())["observable"] == "dist"
    assert captured.out.startswith("analysis complete: tool=reweight")


def test_reweight_with_fes_out(tmp_path, capsys):
    run = meta_run(tmp_path / "run")
    fes_out = tmp_path / "rw_fes.tsv"
    args = ["analysis", "reweight", str(run), "--observable", "dist",
            "--cv", "dist", "--fes-out", str(fes_out), "--bins", "12"]
    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["fes_out"] == str(fes_out)
    lines = fes_out.read_text().splitlines()
    assert lines[0] == "# dist [nm]\tfes [kJ/mol]"
    assert len(lines) == 13
    values = [float(line.split("\t")[-1]) for line in lines[1:]]
    assert all(math.isfinite(v) or math.isinf(v) for v in values)

    # --cv without --fes-out is the documented exit-2 spelling
    assert main(["analysis", "reweight", str(run), "--observable", "dist",
                 "--cv", "dist"]) == 2
    assert "--cv needs --fes-out" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# merge (multi-walker)
# ---------------------------------------------------------------------------


def test_merge_two_walkers_then_analyze_the_merged_dir(tmp_path, capsys):
    walker_a = meta_run(tmp_path / "a", seed=2026)
    walker_b = meta_run(tmp_path / "b", seed=2027)
    merged = tmp_path / "merged"

    assert main(["analysis", "merge", str(walker_a), str(walker_b),
                 "--out", str(merged)]) == 0
    captured = capsys.readouterr().out
    summary = captured.splitlines()[-1]
    assert summary.startswith("analysis complete: tool=merge runs=2")
    payload = json.loads(captured[: captured.rfind("analysis complete:")])
    assert payload["n_hills"] == 20
    assert payload["n_colvar_rows"] == 20
    assert payload["last_step"] == 200
    assert sorted(p.name for p in merged.iterdir()) == \
        ["colvar.tsv", "hills.npz", "manifest.json"]

    # the merged directory is itself a analyzable run dir
    assert main(["analysis", "fes", str(merged)]) == 0
    _, rows = parse_tsv(capsys.readouterr().out)
    assert len(rows) == GRID_BINS
    single = np.array([float(line.split("\t")[-1])
                       for line in (walker_a / "fes.tsv").read_text()
                       .splitlines()[1:]])
    merged_fes = np.array([float(row["fes [kJ/mol]"]) for row in rows])
    # two walkers filled the wells: merged bias is everywhere >= single's
    assert (merged_fes <= single).all()

    # multi-RUN_DIR spellings work without materializing a directory
    assert main(["analysis", "convergence", str(walker_a), str(walker_b),
                 "--blocks", "2"]) == 0
    _, rows = parse_tsv(capsys.readouterr().out)
    assert int(rows[-1]["n_hills"]) == 20


def test_merge_rejects_inconsistent_walkers(tmp_path, capsys):
    walker_a = meta_run(tmp_path / "a", seed=2026)
    other_grid = meta_run(tmp_path / "b", seed=2027)
    config_path = other_grid / "manifest.json"
    manifest = json.loads(config_path.read_text())
    manifest["plan_raw"]["colvars"]["dist"]["bins"] = 21
    config_path.write_text(json.dumps(manifest))
    assert main(["analysis", "merge", str(walker_a), str(other_grid),
                 "--out", str(tmp_path / "m")]) == 2
    assert "biased different grids" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# help surface
# ---------------------------------------------------------------------------


def test_analysis_help_lists_the_five_tools(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["analysis", "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for tool in ("fes", "convergence", "block-average", "reweight", "merge"):
        assert tool in out
