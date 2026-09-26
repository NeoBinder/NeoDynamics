# syntax=docker/dockerfile:1
FROM ghcr.io/prefix-dev/pixi:0.76.2@sha256:8b206ef57005a902cb53f50dbaa47893a4038ca269f0b00038b51f18b1313cd4 AS dependencies
WORKDIR /opt/neodynamics
COPY docker/environment/pixi.toml docker/environment/pixi.lock ./
RUN CONDA_OVERRIDE_CUDA=13.2 pixi install --locked --no-config --concurrent-downloads 4
# Cached PyPI wheels can carry owner-only modes; the runtime is unprivileged.
RUN chmod -R a+rX /opt/neodynamics/.pixi/envs/default

FROM docker.io/library/ubuntu:noble@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90
COPY --from=dependencies /opt/neodynamics/.pixi/envs/default /opt/neodynamics/.pixi/envs/default
RUN groupadd --gid 65532 neomd \
    && useradd --uid 65532 --gid 65532 --home-dir /home/neomd \
       --shell /usr/sbin/nologin --no-create-home neomd
ENV PATH="/opt/neodynamics/.pixi/envs/default/bin:${PATH}" \
    PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOME=/home/neomd NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
ARG WHEEL_PATH
COPY ${WHEEL_PATH} /tmp/neodynamics-install/
RUN python -m pip install --no-deps /tmp/neodynamics-install/*.whl \
    && rm -rf /tmp/neodynamics-install \
    && install -d -o 65532 -g 65532 /home/neomd /var/lib/neodynamics
COPY docker/smoke.py /opt/neodynamics/smoke.py
ARG SOURCE_REVISION
ARG SOURCE_TREE_SHA256
ARG WHEEL_SHA256
ARG SOURCE_VERSION
ARG ENVIRONMENT_LOCK_SHA256
RUN python -c 'import json,sys; json.dump(dict(zip(["version","git_commit","source_tree_sha256","wheel_sha256","environment_lock_sha256"],sys.argv[1:])),open("/opt/neodynamics/provenance.json","w"))' \
    "${SOURCE_VERSION}" "${SOURCE_REVISION}" "${SOURCE_TREE_SHA256}" "${WHEEL_SHA256}" "${ENVIRONMENT_LOCK_SHA256}"
LABEL org.opencontainers.image.title="NeoDynamics resident MD service" \
      org.opencontainers.image.version="${SOURCE_VERSION}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.neobinder.source.tree.sha256="${SOURCE_TREE_SHA256}" \
      org.neobinder.wheel.sha256="${WHEEL_SHA256}" \
      org.neobinder.environment.lock.sha256="${ENVIRONMENT_LOCK_SHA256}"
USER 65532:65532
WORKDIR /var/lib/neodynamics
EXPOSE 8003
ENTRYPOINT ["neomd-server"]
CMD ["serve"]
