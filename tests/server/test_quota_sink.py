"""Output quotas apply to appended tapes, header patches and method-state files."""

import pytest

from neomd_server.sinks import OutputLimitExceeded, QuotaSink


def test_appends_and_header_patches_share_one_output_budget(tmp_path):
    sink = QuotaSink(
        tmp_path / "output", accounting_root=tmp_path, max_bytes=20, direct_write_allowance=5
    )
    sink.write_bytes("state.json", b"12345")
    with sink.binary_writer("output.dcd") as handle:
        handle.write(b"1234567890")
        handle.seek(0)
        handle.write(b"HEAD")
        handle.seek(0, 2)
        with pytest.raises(OutputLimitExceeded):
            handle.write(b"overflow")
    assert sink.read_bytes("output.dcd") == b"HEAD567890"
    sink.write_bytes("state.json", b"1")
    with sink.text_writer("state.tsv") as handle:
        handle.write("abc\n")
        handle.write("def\n")
        with pytest.raises(OutputLimitExceeded):
            handle.write("more\n")
    sink.audit()
    assert sink.used_bytes() == 19


def test_direct_outputs_and_compiled_references_are_counted(tmp_path):
    (tmp_path / "reference.pdbx").write_bytes(b"reference")
    sink = QuotaSink(
        tmp_path / "output", accounting_root=tmp_path, max_bytes=20, direct_write_allowance=10
    )
    path = sink.path("final.pdbx")
    path.write_bytes(b"coordinates")
    sink.audit()
    assert sink.used_bytes() == 20
    with pytest.raises(OutputLimitExceeded):
        sink.write_bytes("extra", b"1")
