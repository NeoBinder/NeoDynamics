-- Immutable exact-byte topology views; historical assets remain unattached.
ALTER TABLE neomd_service.input_operations ADD UNIQUE(scope_id,operation_id,upload_id);
CREATE TABLE neomd_service.topology_views (
    operation_id uuid PRIMARY KEY,
    scope_id uuid NOT NULL, bundle_id uuid NOT NULL,
    owner_id uuid NOT NULL, lease_id uuid NOT NULL,
    binding jsonb NOT NULL CHECK (octet_length(binding::text) <= 4096),
    prepared_schema_version integer NOT NULL CHECK (prepared_schema_version IN (1,2)),
    descriptor bytea CHECK (octet_length(descriptor) BETWEEN 1 AND 1048576),
    descriptor_sha256 text CHECK (descriptor_sha256 ~ '^[0-9a-f]{64}$'),
    view_sha256 text CHECK (view_sha256 ~ '^[0-9a-f]{64}$'),
    inventory_sha256 text CHECK (inventory_sha256 ~ '^[0-9a-f]{64}$'),
    page_count integer NOT NULL DEFAULT 0 CHECK (page_count BETWEEN 0 AND 4096),
    total_page_bytes bigint NOT NULL DEFAULT 0 CHECK (total_page_bytes BETWEEN 0 AND 268435456),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    published_at timestamptz,
    UNIQUE(scope_id,bundle_id,operation_id),
    FOREIGN KEY(scope_id,operation_id,bundle_id)
        REFERENCES neomd_service.input_operations(scope_id,operation_id,upload_id),
    CHECK ((descriptor IS NULL) = (descriptor_sha256 IS NULL)),
    CHECK ((descriptor IS NULL) = (view_sha256 IS NULL)),
    CHECK ((descriptor IS NULL) = (inventory_sha256 IS NULL)),
    CHECK (published_at IS NULL OR descriptor IS NOT NULL)
);
CREATE UNIQUE INDEX topology_published_bundle ON neomd_service.topology_views(scope_id,bundle_id)
    WHERE published_at IS NOT NULL;
CREATE TABLE neomd_service.topology_pages (
    operation_id uuid NOT NULL REFERENCES neomd_service.topology_views(operation_id) ON DELETE CASCADE,
    kind text NOT NULL CHECK (kind IN ('atoms','residues','bonds','particles','topology_particle_indices')),
    page_index integer NOT NULL CHECK (page_index BETWEEN 0 AND 4095),
    stage_sequence integer NOT NULL CHECK (stage_sequence BETWEEN 0 AND 4095),
    first_row bigint NOT NULL CHECK (first_row BETWEEN 0 AND 9007199254740991),
    row_count integer NOT NULL CHECK (row_count BETWEEN 1 AND 512),
    size_bytes integer NOT NULL CHECK (size_bytes BETWEEN 1 AND 1048576),
    content_sha256 text NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    body bytea NOT NULL CHECK (octet_length(body) = size_bytes),
    PRIMARY KEY(operation_id,kind,page_index),
    UNIQUE(operation_id,stage_sequence)
);
ALTER TABLE neomd_service.prepared_assets ADD COLUMN topology_operation_id uuid,
    ADD FOREIGN KEY(scope_id,bundle_id,topology_operation_id)
        REFERENCES neomd_service.topology_views(scope_id,bundle_id,operation_id);
INSERT INTO neomd_service.migrations(version) VALUES (6);
