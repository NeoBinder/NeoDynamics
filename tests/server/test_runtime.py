"""Required host settings and reloadable service credentials do not invent policy."""

import asyncio

import pytest

pytest.importorskip("asyncpg")

from neomd_server.runtime import RuntimeSettings, TokenFile


def environment(tmp_path):
    token = tmp_path / "service-token"
    token.write_text("a" * 32 + "\n")
    values = {
        "PG_DSN": "postgres://fixture:secret@localhost/disposable",
        "TOKEN_FILE": str(token),
        "STORAGE_DIRECTORY": str(tmp_path / "data"),
        "POOL_ID": "test",
        "BIND_HOST": "127.0.0.1",
        "BIND_PORT": "12345",
        "MAX_ACTIVE": "2",
        "MAX_RUNNING_SECONDS": "60",
        "FINALIZATION_TIMEOUT_SECONDS": "30",
        "MAX_QUEUED": "1",
        "QUEUE_TTL_SECONDS": "60",
        "CAPACITY_BYTES": "100000000",
        "MAX_UPLOAD_BYTES": "1000000",
        "MAX_UPLOAD_PART_BYTES": "1000000",
        "MAX_UPLOAD_PARTS": "10",
        "UPLOAD_TTL_SECONDS": "60",
        "MAX_XML_BYTES": "100000",
        "MAX_FORCES": "100",
        "MAX_SYSTEM_PARTICLES": "1000",
        "MAX_BIAS_GRID_POINTS": "1000",
        "MAX_REFERENCE_FRAMES": "2",
        "MAX_REFERENCE_BYTES": "100000",
        "PLATFORM": "cpu",
        "DEVICE_INDEX": "0",
        "CPU_THREADS": "1",
        "STEP_QUANTUM": "4",
        "WORKER_LEASE_MS": "10000",
        "PROBE_TIMEOUT_SECONDS": "30",
        "VERIFICATION_TIMEOUT_SECONDS": "30",
        "MAX_PENDING_INPUT_OPERATIONS": "2",
        "STARTUP_TIMEOUT_SECONDS": "10",
        "BOUNDARY_TIMEOUT_SECONDS": "10",
        "SHUTDOWN_TIMEOUT_SECONDS": "3",
        "MAX_OUTPUT_BYTES": "1000000",
        "OBSERVATION_TTL_SECONDS": "2",
    }
    return {"NEOMD_SERVER_" + key: value for key, value in values.items()}


def test_required_configuration_and_secret_redaction(tmp_path):
    env = environment(tmp_path)
    settings = RuntimeSettings.from_env(env)
    assert settings.admission.max_active == 2
    assert "secret" not in repr(settings) and "service-token" not in repr(settings)
    del env["NEOMD_SERVER_MAX_ACTIVE"]
    del env["NEOMD_SERVER_MAX_QUEUED"]
    del env["NEOMD_SERVER_MAX_RUNNING_SECONDS"]
    del env["NEOMD_SERVER_FINALIZATION_TIMEOUT_SECONDS"]
    del env["NEOMD_SERVER_MAX_PENDING_INPUT_OPERATIONS"]
    del env["NEOMD_SERVER_QUEUE_TTL_SECONDS"]
    with pytest.raises(ValueError) as failure:
        RuntimeSettings.from_env(env)
    assert "MAX_ACTIVE" in str(failure.value) and "QUEUE_TTL_SECONDS" in str(failure.value)
    assert "MAX_QUEUED" in str(failure.value) and "MAX_RUNNING_SECONDS" in str(failure.value)
    assert "FINALIZATION_TIMEOUT_SECONDS" in str(failure.value)
    assert "MAX_PENDING_INPUT_OPERATIONS" in str(failure.value)
    assert "secret" not in str(failure.value)


def test_rotation_reloads_exact_bounded_ascii_credential(tmp_path):
    path = tmp_path / "token"
    token = TokenFile(path)

    async def scenario():
        path.write_text("a" * 32)
        assert await token.verify("a" * 32)
        path.write_text("b" * 32 + "\n")
        assert not await token.verify("a" * 32)
        assert await token.verify("b" * 32)
        for value in ["a" * 31, "a" * 513, "a" * 32 + " ", "a" * 32 + "\n\n", "é" * 32]:
            path.write_text(value)
            assert not await token.verify("b" * 32)
            with pytest.raises(ValueError):
                token.read()

    asyncio.run(scenario())


@pytest.mark.parametrize("value", ["0", "-1", "2147483648", "1.5", ""])
def test_pending_input_capacity_has_no_zero_or_implicit_default(tmp_path, value):
    env = environment(tmp_path)
    env["NEOMD_SERVER_MAX_PENDING_INPUT_OPERATIONS"] = value
    with pytest.raises(ValueError, match="MAX_PENDING_INPUT_OPERATIONS"):
        RuntimeSettings.from_env(env)


def test_cli_reports_closed_missing_configuration_without_secret_values(
    tmp_path, monkeypatch, capsys
):
    import os

    from neomd_server.runtime import main

    for key in os.environ:
        if key.startswith("NEOMD_SERVER_"):
            monkeypatch.delenv(key)
    env = environment(tmp_path)
    del env["NEOMD_SERVER_MAX_ACTIVE"]
    env["NEOMD_SERVER_PG_DSN"] = "invalid://private-user:private-password@private-host/db"
    env["NEOMD_SERVER_TOKEN_FILE"] = "private-relative-token-path"
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(SystemExit) as failure:
        main(["serve"])
    assert failure.value.code == 2
    output = capsys.readouterr().err
    assert "NEOMD_SERVER_MAX_ACTIVE is required" in output
    assert "NEOMD_SERVER_PG_DSN must identify a PostgreSQL database" in output
    assert "NEOMD_SERVER_TOKEN_FILE must be absolute" in output
    for private in ["private-user", "private-password", "private-host", "private-relative"]:
        assert private not in output


def test_cli_keeps_runtime_failures_redacted(tmp_path, monkeypatch, capsys):
    from neomd_server import runtime

    settings = RuntimeSettings.from_env(environment(tmp_path))
    monkeypatch.setattr(RuntimeSettings, "from_env", lambda: settings)

    async def failed_startup(_settings):
        raise ValueError("private runtime path and credential")

    monkeypatch.setattr(runtime, "serve", failed_startup)
    with pytest.raises(SystemExit) as failure:
        runtime.main(["serve"])
    assert failure.value.code == 2
    output = capsys.readouterr().err
    assert "runtime startup failed" in output and "private" not in output


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "2147483648", "true", ""])
@pytest.mark.parametrize('setting', ['MAX_RUNNING_SECONDS', 'FINALIZATION_TIMEOUT_SECONDS'])
def test_runtime_deadlines_require_bounded_explicit_integer(tmp_path, value, setting):
    env = environment(tmp_path)
    env['NEOMD_SERVER_' + setting] = value
    with pytest.raises(ValueError):
        RuntimeSettings.from_env(env)
