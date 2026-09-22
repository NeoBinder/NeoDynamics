CREATE TABLE neomd_service.storage_capacity (
    pool_id text PRIMARY KEY REFERENCES neomd_service.admission(pool_id),
    capacity_bytes bigint NOT NULL CHECK (capacity_bytes > 0)
);
CREATE TABLE neomd_service.asset_uploads (
    upload_id uuid PRIMARY KEY, scope_id uuid NOT NULL, request_id uuid NOT NULL,
    kind text NOT NULL CHECK (kind IN ('prepared','coordinates')),
    pool_id text NOT NULL REFERENCES neomd_service.storage_capacity(pool_id),
    content_sha256 text NOT NULL, request jsonb NOT NULL, receipt jsonb NOT NULL,
    reserved_bytes bigint NOT NULL CHECK (reserved_bytes >= 0),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(scope_id,kind,request_id)
);
CREATE TABLE neomd_service.coordinate_assets (
    reference_id uuid PRIMARY KEY, scope_id uuid NOT NULL, bundle_id uuid NOT NULL,
    upload_id uuid NOT NULL REFERENCES neomd_service.asset_uploads(upload_id),
    metadata jsonb NOT NULL,
    UNIQUE(scope_id,reference_id),
    FOREIGN KEY(scope_id,bundle_id) REFERENCES neomd_service.prepared_assets(scope_id,bundle_id)
);
CREATE TABLE neomd_service.output_reservations (
    session_id uuid PRIMARY KEY REFERENCES neomd_service.sessions(session_id),
    pool_id text NOT NULL REFERENCES neomd_service.storage_capacity(pool_id),
    reserved_bytes bigint NOT NULL CHECK (reserved_bytes > 0)
);
INSERT INTO neomd_service.migrations(version) VALUES (2);
