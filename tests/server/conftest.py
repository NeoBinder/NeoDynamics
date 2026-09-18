"""Explicit disposable PostgreSQL fixture shared by service integration tests."""

import asyncio
import os
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import pytest


@pytest.fixture
def database_dsn():
    asyncpg = pytest.importorskip("asyncpg")
    dsn = os.environ.get("NEOMD_TEST_PG_DSN")
    if not dsn:
        pytest.skip("set NEOMD_TEST_PG_DSN to the explicitly isolated local PG fixture")
    parsed = urlsplit(dsn)
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or parsed.hostname != "127.0.0.1"
        or parsed.port != 5434
        or parsed.path != "/postgres"
        or parsed.query
        or parsed.fragment
    ):
        pytest.fail("test database must use the isolated loopback fixture on port 5434")
    name = "neomd_test_" + uuid4().hex

    async def create():
        connection = await asyncpg.connect(dsn)
        try:
            info = await connection.fetchrow(
                "SELECT current_setting('data_directory') AS data, inet_server_port() AS port"
            )
            if info["data"] != "/tmp/pg-fixture-5434" or info["port"] != 5434:
                pytest.fail("database is not the inspected disposable local fixture")
            await connection.execute(f"CREATE DATABASE {name}")  # locally generated fixed alphabet
        finally:
            await connection.close()

    async def drop():
        connection = await asyncpg.connect(dsn)
        try:
            await connection.execute(f"DROP DATABASE {name}")  # this fixture's unique database only
        finally:
            await connection.close()

    asyncio.run(create())
    try:
        yield urlunsplit(parsed._replace(path="/" + name))
    finally:
        asyncio.run(drop())
