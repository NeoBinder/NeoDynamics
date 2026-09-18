"""Required publication settings, without runtime activation or external I/O."""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

_PREFIX = "NEOMD_SERVER_PUBLICATION_"
_I32_MAX = 2**31 - 1
_PART_MIN = 5 * 1024**2
_PART_MAX = 5 * 1024**3
_RANGES = {
    "max_active": (1, _I32_MAX),
    "max_pending": (1, _I32_MAX),
    "connect_timeout_seconds": (1, _I32_MAX),
    "http_timeout_seconds": (1, _I32_MAX),
    "operation_timeout_seconds": (1, _I32_MAX),
    "poll_interval_seconds": (1, _I32_MAX),
    "lease_ms": (1, 300000),
    "max_part_bytes": (_PART_MIN, _PART_MAX),
}
_FIELDS = ("origin", "provider_id", "token_file", *_RANGES)


class PublicationConfigurationError(ValueError):
    """Collected setting names only; submitted values never enter diagnostics."""

    def __init__(self, names):
        self.names = tuple(names)
        super().__init__("; ".join(self.names))


def _origin(value):
    if (not isinstance(value, str) or not value
            or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
            or any(char in value for char in "?#\\%")):
        return None
    try:
        parsed = urlsplit(value)
        host, port = parsed.hostname, parsed.port
        if (parsed.scheme not in {"http", "https"} or not host
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in {"", "/"} or parsed.netloc.endswith(":")
                or port is not None and not 1 <= port <= 65535):
            return None
        if ":" in host:
            ipaddress.IPv6Address(host)
            if not re.fullmatch(r"\[[^\]]+\](?::[0-9]+)?", parsed.netloc):
                return None
        else:
            ascii_host = host.encode("idna").decode("ascii").removesuffix(".")
            if (len(ascii_host) > 253 or not ascii_host
                    or any(not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label)
                           for label in ascii_host.split("."))):
                return None
        return f"{parsed.scheme}://{parsed.netloc}"
    except (ValueError, UnicodeError):
        return None


def _integer(value):
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        return None
    digits = value.lstrip("0") or "0"
    # All configured maxima fit ten decimal digits; avoid unbounded int parsing.
    return int(digits) if len(digits) <= 10 else None


@dataclass(frozen=True)
class PublicationSettings:
    origin: str
    provider_id: str
    token_file: Path = field(repr=False)
    max_active: int
    max_pending: int
    connect_timeout_seconds: int
    http_timeout_seconds: int
    operation_timeout_seconds: int
    poll_interval_seconds: int
    lease_ms: int
    max_part_bytes: int

    def __post_init__(self):
        invalid = set()
        origin = _origin(self.origin)
        if origin is None:
            invalid.add("origin")
        try:
            if not isinstance(self.provider_id, str) or str(UUID(self.provider_id)) != self.provider_id:
                invalid.add("provider_id")
        except (ValueError, AttributeError):
            invalid.add("provider_id")
        if (not isinstance(self.token_file, Path) or not self.token_file.is_absolute()
                or any(ord(char) < 32 or ord(char) == 127 for char in str(self.token_file))):
            invalid.add("token_file")
        for name, (minimum, maximum) in _RANGES.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                invalid.add(name)
        invalid_numbers = invalid.intersection(_RANGES)
        for lower, upper, strict in (
            ("max_active", "max_pending", False),
            ("connect_timeout_seconds", "http_timeout_seconds", False),
            ("http_timeout_seconds", "operation_timeout_seconds", False),
            ("poll_interval_seconds", "operation_timeout_seconds", True),
        ):
            if lower not in invalid_numbers and upper not in invalid_numbers:
                left, right = getattr(self, lower), getattr(self, upper)
                if left > right or strict and left == right:
                    invalid.update((lower, upper))
        if invalid:
            raise PublicationConfigurationError(_PREFIX + name.upper() for name in _FIELDS if name in invalid)
        object.__setattr__(self, "origin", origin)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> PublicationSettings | None:
        """An absent group is disabled; every field is required once any is set."""
        env = os.environ if environ is None else environ
        if not any(_PREFIX + name.upper() in env for name in _FIELDS):
            return None
        values = {name: env.get(_PREFIX + name.upper()) for name in _FIELDS}
        for name in _RANGES:
            values[name] = _integer(values[name])
        token = values["token_file"]
        if isinstance(token, str) and token:
            values["token_file"] = Path(token)
        return cls(**values)
