-- Durable protein preparation: staged canonical sources and isolated operations.
CREATE TABLE IF NOT EXISTS neomd_service.preparation_policy (
    pool_id text PRIMARY KEY REFERENCES neomd_service.storage_capacity(pool_id),
    max_queued_per_owner integer NOT NULL CHECK (max_queued_per_owner > 0)
);
CREATE TABLE IF NOT EXISTS neomd_service.preparation_source_uploads (
    upload_id uuid PRIMARY KEY,
    scope_id uuid NOT NULL,
    request_id uuid NOT NULL,
    pool_id text NOT NULL REFERENCES neomd_service.storage_capacity(pool_id),
    content_sha256 text NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    request jsonb NOT NULL CHECK (octet_length(request::text) <= 1048576),
    receipt jsonb NOT NULL CHECK (octet_length(receipt::text) <= 1048576),
    reserved_bytes bigint NOT NULL CHECK (reserved_bytes >= 0),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(scope_id,request_id)
);
CREATE TABLE IF NOT EXISTS neomd_service.preparation_operations (
    operation_id uuid PRIMARY KEY,
    scope_id uuid NOT NULL,
    request_id uuid NOT NULL,
    pool_id text NOT NULL REFERENCES neomd_service.preparation_policy(pool_id),
    kind text NOT NULL CHECK (kind = 'protein_preparation'),
    request_sha256 text NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
    request jsonb NOT NULL CHECK (octet_length(request::text) <= 8388608),
    snapshot jsonb NOT NULL CHECK (octet_length(snapshot::text) <= 8388608),
    receipt jsonb CHECK (octet_length(receipt::text) <= 1048576),
    inventory jsonb CHECK (octet_length(inventory::text) <= 1048576),
    qc jsonb CHECK (octet_length(qc::text) <= 1048576),
    source_upload_id uuid NOT NULL
        REFERENCES neomd_service.preparation_source_uploads(upload_id),
    preparation_seed integer NOT NULL CHECK (preparation_seed BETWEEN 1 AND 2147483647),
    state text NOT NULL CHECK (state IN ('queued','running','succeeded','failed','cancelled','interrupted')),
    revision bigint NOT NULL CHECK (revision BETWEEN 1 AND 9007199254740991),
    cancel_requested boolean NOT NULL DEFAULT false,
    owner_id uuid NOT NULL,
    lease_id uuid,
    lease_expires_at timestamptz,
    deadline_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(scope_id,kind,request_id),
    CHECK ((lease_id IS NULL) = (lease_expires_at IS NULL))
);
CREATE INDEX IF NOT EXISTS preparation_pending_owner ON neomd_service.preparation_operations(pool_id,owner_id,created_at)
    WHERE state IN ('queued','running');
CREATE TABLE IF NOT EXISTS neomd_service.preparation_artifacts (
    artifact_id uuid PRIMARY KEY,
    operation_id uuid NOT NULL REFERENCES neomd_service.preparation_operations(operation_id),
    scope_id uuid NOT NULL,
    role text NOT NULL CHECK (role IN
        ('manifest','topology','system','coordinates','mapping','report','source','forcefield','log')),
    path text NOT NULL,
    sha256 text NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(operation_id,path)
);
CREATE INDEX IF NOT EXISTS preparation_artifact_scope ON neomd_service.preparation_artifacts(scope_id,operation_id);
INSERT INTO neomd_service.migrations(version) VALUES (10);
