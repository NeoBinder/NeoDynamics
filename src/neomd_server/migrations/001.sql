CREATE SCHEMA IF NOT EXISTS neomd_service;
CREATE TABLE IF NOT EXISTS neomd_service.migrations (
    version integer PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE neomd_service.admission (
    pool_id text PRIMARY KEY,
    max_active integer NOT NULL CHECK (max_active > 0),
    max_queued integer NOT NULL CHECK (max_queued >= 0),
    queue_ttl_seconds integer NOT NULL CHECK (queue_ttl_seconds > 0)
);
CREATE TABLE neomd_service.prepared_assets (
    bundle_id uuid PRIMARY KEY, scope_id uuid NOT NULL,
    binding jsonb NOT NULL, particle_count bigint NOT NULL CHECK (particle_count >= 0),
    topology_atom_count bigint NOT NULL CHECK (topology_atom_count >= 0),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(), UNIQUE (scope_id, bundle_id)
);
CREATE TABLE neomd_service.protocol_assets (
    protocol_id uuid PRIMARY KEY, scope_id uuid NOT NULL, request_id uuid NOT NULL,
    bundle_id uuid NOT NULL, binding jsonb NOT NULL, method text NOT NULL,
    content_sha256 text NOT NULL, descriptor jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (scope_id, protocol_id), UNIQUE (scope_id, request_id),
    FOREIGN KEY (scope_id,bundle_id) REFERENCES neomd_service.prepared_assets(scope_id,bundle_id)
);
CREATE TABLE neomd_service.sessions (
    session_id uuid PRIMARY KEY, scope_id uuid NOT NULL, request_id uuid NOT NULL,
    payload_sha256 text NOT NULL, pool_id text NOT NULL REFERENCES neomd_service.admission(pool_id),
    bundle_id uuid NOT NULL, protocol_id uuid NOT NULL,
    snapshot jsonb NOT NULL, active_reserved boolean NOT NULL,
    queue_expires_at timestamptz NOT NULL,
    worker_id uuid, worker_lease_expires_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (scope_id,session_id), UNIQUE (scope_id,request_id),
    FOREIGN KEY (scope_id,bundle_id) REFERENCES neomd_service.prepared_assets(scope_id,bundle_id),
    FOREIGN KEY (scope_id,protocol_id) REFERENCES neomd_service.protocol_assets(scope_id,protocol_id)
);
CREATE INDEX sessions_pool ON neomd_service.sessions(pool_id,created_at);
CREATE TABLE neomd_service.controller_requests (
    scope_id uuid NOT NULL, session_id uuid NOT NULL, request_id uuid NOT NULL,
    content_sha256 text NOT NULL, receipt jsonb NOT NULL,
    PRIMARY KEY(scope_id,session_id,request_id),
    FOREIGN KEY(scope_id,session_id) REFERENCES neomd_service.sessions(scope_id,session_id)
);
CREATE TABLE neomd_service.queue_cancellations (
    scope_id uuid NOT NULL, session_id uuid NOT NULL, request_id uuid NOT NULL,
    content_sha256 text NOT NULL, receipt jsonb NOT NULL,
    PRIMARY KEY(scope_id,session_id,request_id),
    FOREIGN KEY(scope_id,session_id) REFERENCES neomd_service.sessions(scope_id,session_id)
);
CREATE TABLE neomd_service.commands (
    scope_id uuid NOT NULL, session_id uuid NOT NULL, command_id uuid NOT NULL,
    request jsonb NOT NULL, receipt jsonb NOT NULL, status text NOT NULL
      CHECK (status IN ('accepted','claimed','applied','rejected','indeterminate')),
    PRIMARY KEY(scope_id,session_id,command_id),
    FOREIGN KEY(scope_id,session_id) REFERENCES neomd_service.sessions(scope_id,session_id)
);
CREATE UNIQUE INDEX command_one_pending ON neomd_service.commands(session_id)
    WHERE status IN ('accepted','claimed');
CREATE TABLE neomd_service.events (
    event_sequence bigserial PRIMARY KEY, scope_id uuid NOT NULL, session_id uuid NOT NULL,
    event jsonb NOT NULL,
    FOREIGN KEY(scope_id,session_id) REFERENCES neomd_service.sessions(scope_id,session_id)
);
CREATE INDEX events_session_cursor ON neomd_service.events(scope_id,session_id,event_sequence);
CREATE TABLE neomd_service.artifacts (
    scope_id uuid NOT NULL, session_id uuid NOT NULL, artifact_id uuid PRIMARY KEY,
    descriptor jsonb NOT NULL, local_ref uuid NOT NULL,
    FOREIGN KEY(scope_id,session_id) REFERENCES neomd_service.sessions(scope_id,session_id)
);
INSERT INTO neomd_service.migrations(version) VALUES (1);
