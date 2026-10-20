"""The engine source tree never imports the extracted service distribution."""

from pathlib import Path

ENGINE_SRC = Path(__file__).resolve().parents[2] / "src" / "neomd"


def test_engine_sources_do_not_import_the_service():
    offenders = [
        path.relative_to(ENGINE_SRC).as_posix()
        for path in sorted(ENGINE_SRC.rglob("*.py"))
        if "neomd_server" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []
