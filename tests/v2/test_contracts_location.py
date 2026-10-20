"""Contract lookup works from a source tree and from installed data files."""

from pathlib import Path

import pytest

from neomd import contracts

ENGINE_SRC = Path(__file__).resolve().parents[2] / "src" / "neomd"


def test_source_tree_contracts_are_found():
    root = contracts.contracts_root()
    assert (root / "molecular-topology/v1/topology.schema.json").is_file()
    directory = contracts.schema_dir("prepared-bundle/v2")
    assert (directory / "manifest.schema.json").is_file()
    paths = list(contracts.schema_paths("prepared-bundle/v1", "prepared-bundle/v2"))
    assert paths
    assert {path.suffix for path in paths} <= {".json", ".yaml", ".yml"}


def test_installed_layout_is_used_when_source_tree_is_absent(tmp_path, monkeypatch):
    installed = tmp_path / "share/neodynamics/contracts/prepared-bundle/v2"
    installed.mkdir(parents=True)
    (installed / "manifest.schema.json").write_text("{}")
    monkeypatch.setattr(contracts, "_source_root", lambda: tmp_path / "missing")
    monkeypatch.setattr(contracts.sysconfig, "get_path", lambda name: str(tmp_path))
    assert contracts.contracts_root() == tmp_path / "share/neodynamics/contracts"
    assert contracts.schema_dir("prepared-bundle/v2") == installed
    assert [path.name for path in contracts.schema_paths("prepared-bundle/v2")] == [
        "manifest.schema.json"
    ]


def test_missing_contracts_report_both_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(contracts, "_source_root", lambda: tmp_path / "source")
    monkeypatch.setattr(contracts, "_installed_root", lambda: tmp_path / "installed")
    with pytest.raises(FileNotFoundError, match="source.*installed"):
        contracts.contracts_root()


def test_contract_lookup_rejects_escaping_relative_paths():
    with pytest.raises(ValueError):
        contracts.schema_dir("../contracts")
    with pytest.raises(ValueError):
        contracts.schema_dir("/etc")
    with pytest.raises(FileNotFoundError):
        contracts.schema_dir("does-not-exist/v1")


def test_engine_sources_do_not_hardcode_the_contract_layout():
    patterns = ('parents[2] / "contracts', "share/neodynamics/contracts")
    offenders = []
    for path in sorted(ENGINE_SRC.rglob("*.py")):
        if path.name == "contracts.py":
            continue
        text = path.read_text(encoding="utf-8")
        if any(pattern in text for pattern in patterns):
            offenders.append(path.relative_to(ENGINE_SRC).as_posix())
    assert offenders == []
