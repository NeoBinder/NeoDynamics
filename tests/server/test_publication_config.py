"""Pure publication configuration never reads credentials or enables a runtime."""

import builtins
import json
import os
import socket
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from neomd_server.publication_config import PublicationConfigurationError, PublicationSettings

PREFIX = "NEOMD_SERVER_PUBLICATION_"


def environment(**changes):
    values = {
        "ORIGIN": "https://publisher.example.test:8443/",
        "PROVIDER_ID": "aabbccdd-1111-2222-3333-444444444444",
        "TOKEN_FILE": "/private-fixture/publisher-token",
        "MAX_ACTIVE": "1",
        "MAX_PENDING": "3",
        "CONNECT_TIMEOUT_SECONDS": "2",
        "HTTP_TIMEOUT_SECONDS": "5",
        "OPERATION_TIMEOUT_SECONDS": "20",
        "POLL_INTERVAL_SECONDS": "1",
        "LEASE_MS": "1000",
        "MAX_PART_BYTES": "5242880",
    }
    values.update(changes)
    return {PREFIX + name: value for name, value in values.items()}


def test_absent_group_is_disabled_and_default_reads_only_environment(monkeypatch):
    assert PublicationSettings.from_env({"NEOMD_SERVER_TOKEN_FILE": "/unrelated"}) is None
    for name in environment():
        monkeypatch.delenv(name, raising=False)
    assert PublicationSettings.from_env() is None
    for name, value in environment().items():
        monkeypatch.setenv(name, value)
    assert PublicationSettings.from_env() == PublicationSettings.from_env(environment())


def test_settings_are_frozen_and_path_is_hidden():
    settings = PublicationSettings.from_env(environment())
    assert settings.origin == "https://publisher.example.test:8443"
    assert settings.provider_id == "aabbccdd-1111-2222-3333-444444444444"
    assert settings.token_file == Path("/private-fixture/publisher-token")
    assert settings.max_active == 1 and settings.max_pending == 3
    assert settings.http_timeout_seconds == 5 and settings.max_part_bytes == 5242880
    assert "private-fixture" not in repr(settings) and "publisher-token" not in repr(settings)
    assert settings.origin in repr(settings) and settings.provider_id in repr(settings)
    with pytest.raises(FrozenInstanceError):
        settings.max_active = 2


@pytest.mark.parametrize("missing", list(environment()))
def test_every_setting_is_required(missing):
    env = environment()
    del env[missing]
    with pytest.raises(PublicationConfigurationError) as error:
        PublicationSettings.from_env(env)
    assert error.value.names == (missing,)


def test_partial_and_empty_groups_collect_all_names_without_values():
    for env in ({PREFIX + "ORIGIN": ""}, {name: "" for name in environment()}):
        with pytest.raises(PublicationConfigurationError) as error:
            PublicationSettings.from_env(env)
        assert error.value.names == tuple(environment())
    env = {name: "private-submitted-value" for name in environment()}
    with pytest.raises(PublicationConfigurationError) as error:
        PublicationSettings.from_env(env)
    assert error.value.names == tuple(environment())
    assert str(error.value) == "; ".join(environment())
    assert "private-submitted-value" not in repr(error.value)


@pytest.mark.parametrize("origin", [
    "http://127.0.0.1:8080", "https://example.test", "https://example.test/",
    "http://[::1]:8080/", "https://publisher.namespace.svc/", "https://localhost/",
])
def test_valid_origins_keep_fixed_transport_path_separate(origin):
    assert PublicationSettings.from_env(environment(ORIGIN=origin)).origin == origin.rstrip("/")


@pytest.mark.parametrize("origin", [
    "", "publisher.example.test", "ftp://host", "https:///", "https://",
    "https://user:secret@host", "https://@host", "https://host?", "https://host#",
    "https://host/?", "https://host/#", "https://host?secret=value", "https://host#secret",
    "https://host/internal/md-publications/v1", "https://host//", "https://host/./",
    " https://host", "https://host ", "https://ho\nst", "https://ho\tst", "https://host\x7f",
    "https://host\x00", "https://host:0", "https://host:65536", "https://host:no",
    "https://host:", "https://host:+80", "https://ho\\st", "https://host%2Fother",
    "https://[::1]suffix", "https://[::1]:", "https://[not-ip]/", "https://[fe80::1%25eth0]",
    "https://host..invalid", "https://-invalid", "https://invalid-", "https://host:80:90",
])
def test_malformed_origin_is_a_closed_name_only_error(origin):
    with pytest.raises(PublicationConfigurationError) as error:
        PublicationSettings.from_env(environment(ORIGIN=origin))
    assert error.value.names == (PREFIX + "ORIGIN",)


@pytest.mark.parametrize("value", [
    "AABBCCDD-1111-2222-3333-444444444444", "aabbccdd111122223333444444444444",
    "{aabbccdd-1111-2222-3333-444444444444}", "not-a-provider", "", " 00000000-0000-0000-0000-000000000000",
])
def test_provider_requires_canonical_uuid(value):
    with pytest.raises(PublicationConfigurationError) as error:
        PublicationSettings.from_env(environment(PROVIDER_ID=value))
    assert error.value.names == (PREFIX + "PROVIDER_ID",)


@pytest.mark.parametrize("path", ["", "relative/token", "~/token", "/private\x00token", "/private\ntoken"])
def test_token_path_is_absolute_without_inspecting_file(path):
    with pytest.raises(PublicationConfigurationError) as error:
        PublicationSettings.from_env(environment(TOKEN_FILE=path))
    assert error.value.names == (PREFIX + "TOKEN_FILE",)


@pytest.mark.parametrize("value", ["0", "-1", "+1", "1.0", "1e3", "NaN", " 1", "1 ", "١", "１", "9" * 10000])
def test_integer_representation_is_bounded_and_ascii(value):
    with pytest.raises(PublicationConfigurationError) as error:
        PublicationSettings.from_env(environment(MAX_ACTIVE=value))
    assert error.value.names == (PREFIX + "MAX_ACTIVE",)


@pytest.mark.parametrize("field", ["MAX_ACTIVE", "MAX_PENDING", "CONNECT_TIMEOUT_SECONDS", "HTTP_TIMEOUT_SECONDS",
                                    "OPERATION_TIMEOUT_SECONDS", "POLL_INTERVAL_SECONDS"])
def test_i32_maxima_reject_overflow(field):
    with pytest.raises(PublicationConfigurationError) as error:
        PublicationSettings.from_env(environment(**{field: "2147483648"}))
    assert error.value.names == (PREFIX + field,)


def test_numeric_boundaries_and_direct_constructor_validation():
    settings = PublicationSettings.from_env(environment(
        MAX_ACTIVE="2147483647", MAX_PENDING="2147483647", CONNECT_TIMEOUT_SECONDS="2147483647",
        HTTP_TIMEOUT_SECONDS="2147483647", OPERATION_TIMEOUT_SECONDS="2147483647",
        POLL_INTERVAL_SECONDS="2147483646", LEASE_MS="300000", MAX_PART_BYTES="5368709120",
    ))
    assert settings.max_active == 2**31 - 1
    assert replace(settings, lease_ms=1).lease_ms == 1
    assert PublicationSettings.from_env(environment(MAX_ACTIVE="0001")).max_active == 1
    for field, value in [("lease_ms", 0), ("lease_ms", 300001), ("max_part_bytes", 5242879),
                         ("max_part_bytes", 5368709121), ("max_active", True), ("lease_ms", 1.0)]:
        with pytest.raises(PublicationConfigurationError) as error:
            replace(settings, **{field: value})
        assert error.value.names == (PREFIX + field.upper(),)


def test_all_independent_cross_field_errors_are_collected():
    with pytest.raises(PublicationConfigurationError) as error:
        PublicationSettings.from_env(environment(MAX_ACTIVE="4", MAX_PENDING="3",
            CONNECT_TIMEOUT_SECONDS="10", HTTP_TIMEOUT_SECONDS="5", OPERATION_TIMEOUT_SECONDS="3",
            POLL_INTERVAL_SECONDS="3"))
    assert error.value.names == tuple(PREFIX + name for name in [
        "MAX_ACTIVE", "MAX_PENDING", "CONNECT_TIMEOUT_SECONDS", "HTTP_TIMEOUT_SECONDS",
        "OPERATION_TIMEOUT_SECONDS", "POLL_INTERVAL_SECONDS",
    ])


def test_part_bounds_match_frozen_contract():
    path = Path(__file__).parents[2] / "contracts/neoapi-md-publication/v1/schema.json"
    bounds = json.loads(path.read_bytes())["$defs"]["Upload"]["properties"]["part_size_bytes"]
    for value in [bounds["minimum"], bounds["maximum"]]:
        assert PublicationSettings.from_env(environment(MAX_PART_BYTES=str(value))).max_part_bytes == value
    for value in [bounds["minimum"] - 1, bounds["maximum"] + 1]:
        with pytest.raises(PublicationConfigurationError):
            PublicationSettings.from_env(environment(MAX_PART_BYTES=str(value)))


def test_parsing_has_no_file_network_or_database_io(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("configuration attempted external I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    assert PublicationSettings.from_env(environment()).max_pending == 3
    assert PublicationSettings.from_env({}) is None
