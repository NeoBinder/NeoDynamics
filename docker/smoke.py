#!/usr/bin/env python3
"""Qualify an installed image on explicitly selected hardware; no service startup."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

METHODS = ("min", "eq", "md", "prod", "metadynamics", "smd", "opes", "gamd", "rbfe")
CONTRACTS = (
    "imd/v1/errors.json", "imd/v1/openapi.json", "imd/v1/protocol.schema.json",
    "scientific-protocol/v1/coordinate-reference.schema.json",
    "scientific-protocol/v1/protocol.schema.json",
    "prepared-bundle/v1/manifest.schema.json", "prepared-bundle/v1/mapping.schema.json",
    "prepared-bundle/v1/source-input.schema.json", "prepared-bundle/v2/manifest.schema.json",
    "prepared-bundle/v2/mapping.schema.json", "prepared-bundle/v2/source-input.schema.json",
    "assembly-report/v1/report.schema.json", "topology-view/v1/page.schema.json",
    "topology-view/v1/view.schema.json", "protein-preparation/v1/catalog.schema.json",
    "trajectory-result/v1/manifest.schema.json", "neoapi-md-publication/v1/openapi.yaml",
    "neoapi-md-publication/v1/schema.json", "neoapi-md-publication/v1/snapshot.json",
    "neoapi-md-publication/v1/README.md",
)
FIXTURE_FILES = (
    "manifest.json", "mapping.json", "system.xml", "topology.pdbx", "coordinates.f64le",
    "sources/0000.pdbx",
)
DEPENDENCY_ROOTS = (
    "NeoDynamics[server]", "OpenMM", "PDBFixer", "openff-toolkit", "openmmforcefields",
)


class SmokeError(RuntimeError):
    """A failed image or hardware gate; never an available capability."""


def read_json(path: Path):
    with path.open("rb") as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise SmokeError(f"JSON exceeds 1 MiB: {path.name}")
    return json.loads(raw)


def check_dependency_closure() -> dict:
    """Validate active Python metadata; Conda's lock owns undeclared dependencies."""
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    environment = default_environment()
    pending = [Requirement(root) for root in DEPENDENCY_ROOTS]
    selected_extras, processed, packages = {}, {}, {}
    while pending:
        requirement = pending.pop()
        name = canonicalize_name(requirement.name)
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            raise SmokeError(f"active dependency missing: {requirement}") from None
        version = distribution.version
        if requirement.url:
            raise SmokeError(f"active direct-URL dependency needs an explicit identity check: {name}")
        if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
            raise SmokeError(f"active dependency mismatch: {requirement}; installed {version}")
        extras = selected_extras.setdefault(name, set())
        extras.update(canonicalize_name(extra) for extra in requirement.extras)
        provided = {canonicalize_name(extra)
                    for extra in distribution.metadata.get_all("Provides-Extra", [])}
        if extras - provided:
            raise SmokeError(f"active dependency requests undeclared extras: {name}[{','.join(sorted(extras - provided))}]")
        if name in processed and processed[name] == extras:
            continue
        processed[name] = set(extras)
        declared = distribution.requires
        active, inactive = [], []
        for raw in declared or ():
            dependency = Requirement(raw)
            if dependency.marker is not None and not any(
                dependency.marker.evaluate({**environment, "extra": extra})
                for extra in {"", *extras}
            ):
                inactive.append(raw)
                continue
            active.append(raw)
            pending.append(dependency)
        packages[name] = {
            "version": version, "extras": sorted(extras), "active_requires_dist": active,
            "inactive_requires_dist": inactive,
            "requires_dist_declared": declared is not None,
        }
    return {
        "roots": list(DEPENDENCY_ROOTS), "marker_environment": environment,
        "packages": dict(sorted(packages.items())),
        "scope": "Active recursive Requires-Dist only; Conda lock remains authoritative for Conda dependencies, including packages with null Python metadata.",
    }


def check_installation(provenance_path: Path) -> dict:
    """Use wheel metadata/data locations, never a checkout or version fallback."""
    distribution = importlib.metadata.distribution("NeoDynamics")
    version = distribution.version
    provenance = read_json(provenance_path)
    if not isinstance(provenance, dict) or provenance.get("version") != version:
        raise SmokeError("provenance version does not match installed NeoDynamics")
    revision = provenance.get("git_commit")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SmokeError("provenance git_commit must be a full lowercase Git SHA1")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    if direct_url.get("dir_info", {}).get("editable"):
        raise SmokeError("image qualification requires an installed wheel, not an editable checkout")
    records = {Path(distribution.locate_file(item)).resolve()
               for item in distribution.files or ()}
    origins = {}
    for module in ("neomd", "neomd_server"):
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() not in records:
            raise SmokeError(f"{module} is not imported from the installed distribution")
        origins[module] = str(Path(spec.origin).resolve())
    if importlib.import_module("neomd").__version__ != version:
        raise SmokeError("neomd runtime version differs from distribution metadata")
    versions = {name: importlib.metadata.version(name) for name in ("OpenMM", "pdbfixer")}
    if versions["OpenMM"] not in {"8.6", "8.6.0"}:
        raise SmokeError("image requires exact OpenMM 8.6.0")
    if versions["pdbfixer"] not in {"1.12", "1.12.0"}:
        raise SmokeError("image requires exact PDBFixer 1.12.0")
    # Check that the distribution is importable, not merely present as metadata.
    importlib.import_module("openmm")
    importlib.import_module("pdbfixer")
    server = Path(origins["neomd_server"]).parent
    migrations = server / "migrations"
    if {path.name for path in migrations.glob("*.sql")} != {f"{n:03d}.sql" for n in range(1, 10)}:
        raise SmokeError("installed migrations must contain exactly 001.sql through 009.sql")
    contracts = Path(sysconfig.get_path("data")) / "share/neodynamics/contracts"
    assets = [(f"migrations/{n:03d}.sql", migrations / f"{n:03d}.sql") for n in range(1, 10)]
    assets += [(f"contracts/{name}", contracts / name) for name in CONTRACTS]
    assets += [(f"fixture/{name}", server / "fixtures/qualification" / name) for name in FIXTURE_FILES]
    inventory = {}
    for label, path in assets:
        if path.resolve() not in records or not path.is_file():
            raise SmokeError(f"required asset missing from installed wheel: {label}")
        raw = path.read_bytes()
        if not raw:
            raise SmokeError(f"empty installed asset: {label}")
        if path.suffix == ".json":
            json.loads(raw)
        inventory[label] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    closure = check_dependency_closure()
    return {"version": version, "git_commit": revision, "dependencies": versions,
            "dependency_closure": closure,
            "python": sys.version.split()[0], "module_origins": origins, "assets": inventory}


def check_methods(result, platform: str) -> None:
    if not isinstance(result, dict) or set(result) != set(METHODS):
        raise SmokeError("qualification did not return exactly the expected nine methods")
    for method in METHODS:
        item = result[method]
        if (not isinstance(item, dict) or item.get("availability") != "available"
                or item.get("qualification") != platform or item.get("reason") is not None
                or not isinstance(item.get("controls"), dict)):
            raise SmokeError(f"method qualification failed: {method}")


def worker(args) -> None:
    receipt = check_installation(args.provenance)
    from neomd_server.qualification import METHODS as implemented_methods
    from neomd_server.qualification import probe_methods

    if tuple(implemented_methods) != METHODS:
        raise SmokeError("public probe method inventory changed; review the smoke gate")
    work = args._worker_root / "work"
    work.mkdir()
    result = probe_methods(platform=args.platform, device_index=args.device_index,
                           cpu_threads=args.cpu_threads, work_directory=str(work))
    if list(work.iterdir()):
        raise SmokeError("public probe left scientific scratch after returning")
    receipt.update(platform=args.platform, device_index=args.device_index,
                   cpu_threads=args.cpu_threads, methods=result)
    # Preserve unavailable results for the supervising process's failure report.
    (args._worker_root / "result.json").write_text(json.dumps(receipt, allow_nan=False) + "\n")
    check_methods(result, args.platform)


def interrupted(signum, _frame):
    raise InterruptedError(f"smoke interrupted by signal {signum}")


def run(args) -> dict:
    """Success requires both a complete result and a reaped, zero-exit child."""
    signal.signal(signal.SIGTERM, interrupted)
    root = Path(tempfile.mkdtemp(prefix="neodynamics-image-smoke-")).resolve()
    process = None
    try:
        command = [sys.executable, "-I", str(Path(__file__).resolve()),
                   "--platform", args.platform, "--device-index", args.device_index,
                   "--cpu-threads", str(args.cpu_threads), "--timeout", str(args.timeout),
                   "--provenance", str(args.provenance), "--_worker-root", str(root)]
        with (root / "child.log").open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt, InterruptedError):
                process.kill()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    raise SmokeError(f"child closure unconfirmed; scratch retained at {root}") from None
                raise SmokeError("qualification timed out/interrupted; child killed and reaped") from None
        receipt_path = root / "result.json"
        receipt = read_json(receipt_path) if receipt_path.is_file() else {}
        if code != 0:
            # Tiny trusted fixture logs only; never print environment/credentials.
            with (root / "child.log").open("rb") as log:
                log.seek(max(0, (root / "child.log").stat().st_size - 8192))
                detail = log.read().decode("utf-8", errors="replace")
            print(detail, file=sys.stderr, end="")
            if receipt.get("methods"):
                print(json.dumps({"methods": receipt["methods"]}), file=sys.stderr)
            raise SmokeError(f"qualification child exited with status {code}")
        check_methods(receipt.get("methods"), args.platform)
        if list((root / "work").iterdir()):
            raise SmokeError("scientific scratch remained after child exit")
        return {"result": "passed", "child_exit_code": code, "child_reaped": True,
                "timeout_seconds": args.timeout, **receipt}
    finally:
        if process is None or process.poll() is not None:
            shutil.rmtree(root)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--device-index", required=True, help="one container-visible device index; use 0 for CPU")
    parser.add_argument("--timeout", type=float, required=True, help="wall-clock seconds for the entire child")
    parser.add_argument("--cpu-threads", type=int, default=1, help="smoke-only CPU threads (default: 1)")
    parser.add_argument("--provenance", type=Path, required=True, help="image JSON with version and git_commit")
    parser.add_argument("--_worker-root", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9]{1,10}", args.device_index):
        parser.error("--device-index requires one nonnegative decimal index")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if not 1 <= args.cpu_threads <= 1024:
        parser.error("--cpu-threads must be between 1 and 1024")
    args.provenance = args.provenance.resolve()
    return args


def main(argv=None) -> int:
    args = arguments(argv)
    try:
        if args._worker_root is not None:
            worker(args)
        else:
            print(json.dumps(run(args), allow_nan=False))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"result": "failed", "error": f"{type(error).__name__}: {error}"}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
