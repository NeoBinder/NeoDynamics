-- Historical input rows have no inferred verification/projection proof.
ALTER TABLE neomd_service.prepared_assets
    ADD COLUMN particle_identity_sha256 text,
    ADD COLUMN preview_projection_sha256 text,
    ADD CONSTRAINT prepared_identity_hash CHECK
        (particle_identity_sha256 IS NULL OR particle_identity_sha256 ~ '^[0-9a-f]{64}$'),
    ADD CONSTRAINT prepared_projection_hash CHECK
        (preview_projection_sha256 IS NULL OR preview_projection_sha256 ~ '^[0-9a-f]{64}$'),
    ADD CONSTRAINT prepared_projection_pair CHECK
        ((particle_identity_sha256 IS NULL) = (preview_projection_sha256 IS NULL));
ALTER TABLE neomd_service.asset_uploads ADD UNIQUE(scope_id,upload_id);

CREATE TABLE neomd_service.input_policy (
    pool_id text PRIMARY KEY REFERENCES neomd_service.storage_capacity(pool_id),
    max_pending_operations integer NOT NULL CHECK (max_pending_operations > 0)
);
CREATE TABLE neomd_service.input_operations (
    operation_id uuid PRIMARY KEY, scope_id uuid NOT NULL, request_id uuid NOT NULL,
    pool_id text NOT NULL REFERENCES neomd_service.input_policy(pool_id),
    kind text NOT NULL CHECK (kind IN ('prepared_commit','coordinate_commit','protocol_register')),
    request_sha256 text NOT NULL CHECK (request_sha256 ~ '^[0-9a-f]{64}$'),
    request jsonb NOT NULL CHECK (octet_length(request::text) <= 1048576),
    source_snapshot jsonb NOT NULL CHECK (octet_length(source_snapshot::text) <= 1048576),
    upload_id uuid, protocol_id uuid,
    state text NOT NULL CHECK (state IN ('accepted','running','succeeded','failed','interrupted')),
    revision bigint NOT NULL CHECK (revision BETWEEN 1 AND 9007199254740991),
    snapshot jsonb NOT NULL CHECK (octet_length(snapshot::text) <= 1048576),
    owner_id uuid NOT NULL, lease_id uuid, lease_expires_at timestamptz,
    active_reserved boolean NOT NULL DEFAULT false,
    deadline_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(scope_id,kind,request_id),
    FOREIGN KEY(scope_id,upload_id) REFERENCES neomd_service.asset_uploads(scope_id,upload_id),
    CHECK ((kind='protocol_register' AND upload_id IS NULL AND protocol_id IS NOT NULL)
        OR (kind<>'protocol_register' AND upload_id IS NOT NULL AND protocol_id IS NULL)),
    CHECK ((lease_id IS NULL) = (lease_expires_at IS NULL))
);
CREATE UNIQUE INDEX input_one_active_upload ON neomd_service.input_operations(scope_id,kind,upload_id)
    WHERE upload_id IS NOT NULL AND state IN ('accepted','running');
CREATE INDEX input_pending_owner ON neomd_service.input_operations(pool_id,owner_id,created_at)
    WHERE state IN ('accepted','running');
INSERT INTO neomd_service.migrations(version) VALUES (5);
