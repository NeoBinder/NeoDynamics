"""Explicit host configuration and the independent service runtime assembly."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import ipaddress
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from .assets import InputStore, UploadLimits
from .ledger import AdmissionLimits, Ledger
from .protocols import ScientificLimits
from .publication_config import PublicationSettings
from .publication_runtime import PublicationCoordinator
from .supervisor import ExecutionLimits, Supervisor

_TOKEN = re.compile(r"[A-Za-z0-9._~-]{32,512}\Z", re.ASCII)


class ConfigurationError(ValueError):
    """Closed host setting diagnostics containing names/constraints, never values."""


@dataclass(frozen=True)
class TokenFile:
    path: Path = field(repr=False)

    def read(self):
        try:
            with self.path.open("rb") as handle:
                data = handle.read(514)
            token = data.decode("ascii").removesuffix("\n")
        except (OSError, UnicodeError):
            raise ValueError("service credential file is unreadable or invalid") from None
        if not _TOKEN.fullmatch(token):
            raise ValueError("service credential must be 32–512 supported ASCII characters")
        return token

    async def verify(self, supplied):
        if not isinstance(supplied, str) or not _TOKEN.fullmatch(supplied):
            return False
        try:
            expected = await asyncio.to_thread(self.read)
        except ValueError:
            return False
        return hmac.compare_digest(expected, supplied)


@dataclass(frozen=True)
class RuntimeSettings:
    dsn: str = field(repr=False)
    token_file: Path = field(repr=False)
    storage_directory: Path
    pool_id: str
    bind_host: str
    bind_port: int
    admission: AdmissionLimits
    uploads: UploadLimits
    science: ScientificLimits
    execution: ExecutionLimits
    verification_timeout_seconds: int
    max_pending_input_operations: int

    @classmethod
    def from_env(cls, environ=None):
        env = os.environ if environ is None else environ
        problems = []

        def read(name):
            key = "NEOMD_SERVER_" + name
            value = env.get(key)
            if not value:
                problems.append(key + " is required")
                return ""
            return value

        def integer(name):
            value = read(name)
            if not value.isascii() or not value.isdecimal() or int(value) > 2**53 - 1:
                problems.append("NEOMD_SERVER_" + name + " must be a safe integer")
                return 0
            return int(value)

        dsn, token, directory, pool, host = [
            read(name)
            for name in ["PG_DSN", "TOKEN_FILE", "STORAGE_DIRECTORY", "POOL_ID", "BIND_HOST"]
        ]
        port = integer("BIND_PORT")
        verification_timeout_seconds = integer("VERIFICATION_TIMEOUT_SECONDS")
        if verification_timeout_seconds < 1:
            problems.append("NEOMD_SERVER_VERIFICATION_TIMEOUT_SECONDS must be positive")
        max_pending_input_operations = integer("MAX_PENDING_INPUT_OPERATIONS")
        if not 1 <= max_pending_input_operations <= 2**31 - 1:
            problems.append("NEOMD_SERVER_MAX_PENDING_INPUT_OPERATIONS must be in 1..2147483647")
        admission_values = [
            integer(name) for name in ["MAX_ACTIVE", "MAX_QUEUED", "QUEUE_TTL_SECONDS"]
        ]
        upload_values = [
            integer(name)
            for name in [
                "CAPACITY_BYTES",
                "MAX_UPLOAD_BYTES",
                "MAX_UPLOAD_PART_BYTES",
                "MAX_UPLOAD_PARTS",
                "UPLOAD_TTL_SECONDS",
                "MAX_XML_BYTES",
                "MAX_FORCES",
                "MAX_SYSTEM_PARTICLES",
            ]
        ]
        science_values = [
            integer(name)
            for name in ["MAX_BIAS_GRID_POINTS", "MAX_REFERENCE_FRAMES", "MAX_REFERENCE_BYTES"]
        ]
        execution_values = [
            read("PLATFORM"),
            read("DEVICE_INDEX"),
            *[
                integer(name)
                for name in [
                    "CPU_THREADS",
                    "STEP_QUANTUM",
                    "WORKER_LEASE_MS",
                    "PROBE_TIMEOUT_SECONDS",
                    "STARTUP_TIMEOUT_SECONDS",
                    "BOUNDARY_TIMEOUT_SECONDS",
                    "SHUTDOWN_TIMEOUT_SECONDS",
                    "MAX_OUTPUT_BYTES",
                    "OBSERVATION_TTL_SECONDS",
                    "MAX_RUNNING_SECONDS",
                    "FINALIZATION_TIMEOUT_SECONDS",
                ]
            ],
        ]
        try:
            parsed = urlsplit(dsn)
            if (
                parsed.scheme not in {"postgres", "postgresql"}
                or not parsed.hostname
                or parsed.path in {"", "/"}
            ):
                raise ValueError()
        except ValueError:
            problems.append("NEOMD_SERVER_PG_DSN must identify a PostgreSQL database")
        for name, value in [("TOKEN_FILE", token), ("STORAGE_DIRECTORY", directory)]:
            if not Path(value).is_absolute():
                problems.append("NEOMD_SERVER_" + name + " must be absolute")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", pool):
            problems.append("NEOMD_SERVER_POOL_ID must be a bounded identifier")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            problems.append("NEOMD_SERVER_BIND_HOST must be an IP address")
        if not 1 <= port <= 65535:
            problems.append("NEOMD_SERVER_BIND_PORT must be in 1..65535")
        groups = []
        for constructor, values in [
            (AdmissionLimits, admission_values),
            (UploadLimits, upload_values),
            (ScientificLimits, science_values),
            (ExecutionLimits, execution_values),
        ]:
            try:
                groups.append(constructor(*values))
            except (TypeError, ValueError) as error:
                problems.append(str(error))
        if problems:
            raise ConfigurationError("; ".join(problems))
        admission, uploads, science, execution = groups
        if execution.max_output_bytes > uploads.capacity_bytes:
            raise ConfigurationError("output reservation exceeds total storage capacity")
        try:
            # Startup verifies the mounted credential; requests reload rotations.
            TokenFile(Path(token)).read()
        except ValueError as error:
            raise ConfigurationError("NEOMD_SERVER_TOKEN_FILE: " + str(error)) from None
        return cls(
            dsn,
            Path(token),
            Path(directory),
            pool,
            host,
            port,
            admission,
            uploads,
            science,
            execution,
            verification_timeout_seconds,
            max_pending_input_operations,
        )


def publication_coordinator(ledger, inputs, storage_directory, *, pool_id, shutdown_timeout_seconds):
    """Explicit activation only: absent publication settings leave it disabled."""
    publication = PublicationSettings.from_env()
    if publication is None:
        return None
    return PublicationCoordinator(
        ledger, inputs, pool_id=pool_id, settings=publication,
        shutdown_timeout_seconds=shutdown_timeout_seconds,
    )


async def serve(settings):
    import uvicorn

    from .app import TransportSettings, create_app

    ledger = await Ledger.connect(settings.dsn)
    try:
        inputs = InputStore(
            ledger,
            settings.storage_directory,
            pool_id=settings.pool_id,
            limits=settings.uploads,
            scientific_limits=settings.science,
            verification_timeout_seconds=settings.verification_timeout_seconds,
            max_pending_input_operations=settings.max_pending_input_operations,
            shutdown_timeout_seconds=settings.execution.shutdown_timeout_seconds,
        )
        transport = TransportSettings(
            TokenFile(settings.token_file).verify,
            settings.uploads.max_upload_part_bytes,
            min(settings.uploads.max_upload_bytes, settings.science.max_reference_bytes),
        )
        supervisor = Supervisor(
            ledger,
            inputs,
            pool_id=settings.pool_id,
            admission=settings.admission,
            execution=settings.execution,
            max_auth_staleness_seconds=int(
                transport.auth_interval_seconds + transport.auth_timeout_seconds
            ),
            publication=publication_coordinator(
                ledger, inputs, settings.storage_directory, pool_id=settings.pool_id,
                shutdown_timeout_seconds=settings.execution.shutdown_timeout_seconds),
        )
        app = create_app(ledger, inputs, supervisor, transport)
        config = uvicorn.Config(
            app,
            host=settings.bind_host,
            port=settings.bind_port,
            workers=1,
            ws="websockets",
            ws_max_size=8388608,
            ws_max_queue=1,
            ws_per_message_deflate=False,
            proxy_headers=False,
            access_log=False,
            timeout_graceful_shutdown=settings.execution.shutdown_timeout_seconds,
        )
        await uvicorn.Server(config).serve()
    finally:
        await ledger.close()


async def migrate(settings):
    ledger = await Ledger.connect(settings.dsn)
    try:
        await ledger.migrate()
    finally:
        await ledger.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="neomd-server", description="Independent NeoDynamics service"
    )
    parser.add_argument("operation", choices=["serve", "migrate"])
    arguments = parser.parse_args(argv)
    try:
        settings = RuntimeSettings.from_env()
    except ConfigurationError as error:
        parser.exit(2, "service configuration failed: " + str(error) + "\n")
    except (ValueError, OSError):
        parser.exit(2, "service configuration failed\n")
    try:
        asyncio.run(serve(settings) if arguments.operation == "serve" else migrate(settings))
    except (ValueError, OSError):
        parser.exit(
            2, "service runtime startup failed; inspect host diagnostics\n"
        )


if __name__ == "__main__":
    main()
