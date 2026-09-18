#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
image="${1:?Usage: docker/build.sh IMAGE [VERIFIED_RUNTIME_BASE_AT_DIGEST]}"
base_image="${2:-}"
mkdir -p docker/.build
wheel_directory="$(mktemp -d docker/.build/wheel.XXXXXX)"
uv build --wheel --out-dir "${wheel_directory}"
wheels=("${wheel_directory}"/*.whl)
[[ ${#wheels[@]} -eq 1 && -f "${wheels[0]}" ]]
revision="$(git rev-parse HEAD)"
wheel_hash="$(sha256sum "${wheels[0]}" | cut -d ' ' -f1)"
lock_hash="$(sha256sum docker/environment/pixi.lock | cut -d ' ' -f1)"
version="$(python3 - "${wheels[0]}" <<'PY'
import email
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as wheel:
    metadata = [name for name in wheel.namelist() if name.endswith('.dist-info/METADATA')]
    assert len(metadata) == 1
    print(email.message_from_bytes(wheel.read(metadata[0]))['Version'])
PY
)"
source_hash="$(python3 - <<'PY'
import hashlib
import pathlib
import subprocess

paths = subprocess.check_output([
    'git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z',
]).split(b'\0')
digest = hashlib.sha256()
for name in sorted(set(paths) - {b''}):
    path = pathlib.Path(name.decode())
    if path.is_file():
        digest.update(name + b'\0' + hashlib.sha256(path.read_bytes()).digest())
print(digest.hexdigest())
PY
)"
dockerfile=docker/Dockerfile
build_args=()
if [[ -n "${base_image}" ]]; then
    [[ "${base_image}" == *@sha256:* ]] || { echo 'Runtime base must use a registry digest' >&2; exit 2; }
    podman pull "${base_image}"
    base_lock="$(podman image inspect "${base_image}" --format '{{index .Config.Labels "org.neobinder.environment.lock.sha256"}}')"
    [[ "${base_lock}" == "${lock_hash}" ]] || { echo 'Dependency lock changed; a full build is required' >&2; exit 2; }
    dockerfile=docker/Dockerfile.source
    build_args+=(--build-arg "BASE_IMAGE=${base_image}")
fi
podman build --network host -f "${dockerfile}" -t "${image}" \
    --build-arg "WHEEL_PATH=${wheels[0]}" \
    --build-arg "SOURCE_REVISION=${revision}" \
    --build-arg "SOURCE_TREE_SHA256=${source_hash}" \
    --build-arg "SOURCE_VERSION=${version}" \
    --build-arg "ENVIRONMENT_LOCK_SHA256=${lock_hash}" \
    --build-arg "WHEEL_SHA256=${wheel_hash}" "${build_args[@]}" .
printf 'source_revision=%s\nsource_tree_sha256=%s\nwheel_sha256=%s\n' \
    "${revision}" "${source_hash}" "${wheel_hash}"
