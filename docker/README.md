# NeoDynamics container images

Build from the owning task worktree root. The portable container environment is
`docker/environment/pixi.toml` plus its locked dependencies; it is separate from
the root development environment. The image installs a versioned wheel, packaged
contracts/migrations and `docker/smoke.py`, then runs `neomd-server serve` as
UID/GID 65532 on port 8003. The server's required runtime settings and secrets are
supplied at deployment, not baked into the image.

## Full and source-only builds

```bash
# First runtime or an explicitly changed dependency lock:
bash docker/build.sh registry.example/neodynamics:dev

# Source-only change: reuse a verified, published runtime at its exact digest.
bash docker/build.sh registry.example/neodynamics:dev \
  registry.example/neodynamics@sha256:REPLACE_WITH_VERIFIED_RUNTIME_DIGEST
```

`build.sh` builds one wheel with `uv build`, derives its version from package
metadata, and records Git revision, actual tracked/nonignored source-tree hash,
wheel SHA256 and container environment-lock SHA256. The source hash accounts for
dirty source content; HEAD alone is not the build identity. Ignored wheel/build
artifacts stay under `docker/.build/`. Record the exact source inventory when
other owners are editing the same worktree; later documentation changes do not
retroactively belong to an earlier image's source hash.

The full Dockerfile resolves the pinned dedicated Pixi environment into the
runtime image. `Dockerfile.source` pulls the supplied digest, requires its recorded
environment-lock hash to match, and force-reinstalls only the new wheel without
resolving dependencies. A changed lock requires a reviewed full runtime build.
Both images register the runtime UID/GID 65532 as `neomd` so installed Python
toolkits can resolve the nonroot account inside Kubernetes.
Both paths use `podman build --network host`; workstation proxies/credentials are
not portable Dockerfile configuration. Keep container lock, OpenMM pin and driver
CUDA compatibility aligned; do not copy a newer development CUDA lock blindly.

The current container environment selects CUDA 13.2, OpenMM 8.6, NumPy 2.5.2 and
PDBFixer 1.12. A matching driver tier and actual GPU smoke are required. These
pins are packaging inputs, not permission to change engine science defaults.

## Image qualification and provenance

The current Dockerfiles/build script own dependency validation. Do not claim a
successful whole-environment `pip check`: Conda AmberTools optional scripts can
declare older NumPy metadata incompatible with this pinned NumPy version.
Preserve the approved lock while the packaging owner validates the direct/runtime
dependency closure and actual imported runtime. An optional-script metadata
conflict neither proves the scientific runtime is broken nor qualifies it.

After the selected image's build gates pass, run its independent bounded smoke:

```bash
podman run --rm --network host --entrypoint python \
  registry.example/neodynamics:dev /opt/neodynamics/smoke.py \
  --platform cpu --device-index 0 --timeout 180 \
  --provenance /opt/neodynamics/provenance.json
```

Run the corresponding `--platform cuda --device-index 0` smoke with the selected
host's configured GPU access, or in the operator's NVIDIA-runtime qualification
pod. Do not assume `--device-index 0` attaches a GPU to an ordinary container.
The smoke CLI and result fields are authoritative in `smoke.py`; record the
actual exit/result, platform and image identity. CPU success is not CUDA evidence.

`/opt/neodynamics/provenance.json` stores version, Git commit, source-tree SHA256,
wheel SHA256 and environment-lock SHA256, with corresponding image labels. Compare
these with build evidence and the smoke result before publishing. Push with the
existing registry credential mechanism and record the **registry manifest digest**
returned by the push, then verify the remote reference and content. A local
image/config ID is not the registry manifest digest. Reuse this tested digest for
the service and its explicit migration Job.

Build/smoke/push evidence does not establish workload readiness, scientific scale
or remote publication. Deployment templates, exact Secret interfaces and the
operator's database/rollout sequence are in [deploy/README.md](../deploy/README.md).
No image build runs migrations or activates publication/cleanup automatically.
