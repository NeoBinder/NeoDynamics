-- Deletion intents and lifetime/inventory ownership survive scientific rows.
-- No historical process/reader closure is inferred or backfilled.
CREATE TABLE neomd_service.session_deletions (
    operation_id uuid PRIMARY KEY,
    scope_id uuid NOT NULL,
    request_id uuid NOT NULL,
    request_sha256 text NOT NULL CHECK(request_sha256 ~ '^[0-9a-f]{64}$'),
    request jsonb NOT NULL,
    snapshot jsonb NOT NULL,
    stage text NOT NULL CHECK(stage IN ('accepted','quiescing','quiesced',
        'removing_outputs','removing_records','succeeded','blocked','failed','rejected')),
    revision bigint NOT NULL CHECK(revision BETWEEN 1 AND 9007199254740991),
    deletion_fence bigint NOT NULL CHECK(deletion_fence BETWEEN 0 AND 9007199254740991),
    effects_started boolean NOT NULL,
    inventory_id uuid,
    inventory_sha256 text CHECK(inventory_sha256 ~ '^[0-9a-f]{64}$'),
    removal_request_id uuid,
    removal_request_sha256 text CHECK(removal_request_sha256 ~ '^[0-9a-f]{64}$'),
    public_cleanup_barrier jsonb,
    owner_id uuid,
    lease_id uuid,
    lease_expires_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(scope_id,request_id),
    UNIQUE(scope_id,operation_id),
    CHECK((inventory_id IS NULL) = (inventory_sha256 IS NULL)),
    CHECK((removal_request_id IS NULL) = (removal_request_sha256 IS NULL)),
    CHECK((stage='rejected' AND NOT effects_started AND deletion_fence=0)
        OR (stage<>'rejected' AND effects_started AND deletion_fence>0))
);
CREATE INDEX session_deletions_pending ON neomd_service.session_deletions(stage,created_at)
    WHERE stage NOT IN ('succeeded','rejected');

-- Only admitted deletions populate this gate. Rejected attempts never fence a
-- session. Original identity survives physical removal, including unsent creates.
CREATE TABLE neomd_service.session_retirements (
    scope_id uuid NOT NULL,
    session_id uuid NOT NULL,
    create_request_id uuid NOT NULL,
    create_request_sha256 text CHECK(create_request_sha256 ~ '^[0-9a-f]{64}$'),
    operation_id uuid NOT NULL,
    expected_service_revision bigint CHECK(expected_service_revision BETWEEN 1 AND 9007199254740991),
    deletion_fence bigint NOT NULL CHECK(deletion_fence BETWEEN 1 AND 9007199254740991),
    state text NOT NULL CHECK(state IN ('deleting','deleted')),
    PRIMARY KEY(scope_id,session_id),
    UNIQUE(scope_id,create_request_id),
    FOREIGN KEY(scope_id,operation_id) REFERENCES neomd_service.session_deletions(scope_id,operation_id)
);

-- Register before granting a child, FD, stream or filesystem thread. A lifetime
-- can read multiple sessions; closure is the joined owner proof, never expiry.
CREATE TABLE neomd_service.session_lifetimes (
    lifetime_id uuid PRIMARY KEY,
    scope_id uuid NOT NULL,
    kind text NOT NULL CHECK(kind IN ('engine','finalizer','uploader','analysis',
        'continuation_reader','preview_transport','event_transport','cleanup_thread')),
    state text NOT NULL CHECK(state IN ('admitted','active','closing','closed','unknown')),
    owner_id uuid,
    owner_boot_id uuid,
    replica_id text,
    process_id integer CHECK(process_id>0),
    process_started bigint CHECK(process_started>=0),
    related_operation_id uuid,
    admitted_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    closed_at timestamptz,
    closure_kind text CHECK(closure_kind IN ('joined_process','closed_transport','joined_thread','trusted_owner_teardown')),
    closure_receipt jsonb,
    CHECK(state='unknown' OR (owner_id IS NOT NULL AND owner_boot_id IS NOT NULL
        AND replica_id IS NOT NULL AND length(replica_id)>0)),
    CHECK((state='closed') = (closed_at IS NOT NULL AND closure_kind IS NOT NULL AND closure_receipt IS NOT NULL)),
    UNIQUE(scope_id,lifetime_id)
);
CREATE TABLE neomd_service.session_lifetime_members (
    scope_id uuid NOT NULL,
    lifetime_id uuid NOT NULL,
    session_id uuid NOT NULL,
    PRIMARY KEY(lifetime_id,session_id),
    FOREIGN KEY(scope_id,lifetime_id) REFERENCES neomd_service.session_lifetimes(scope_id,lifetime_id) ON DELETE CASCADE
);
CREATE INDEX session_lifetime_members_gate ON neomd_service.session_lifetime_members(scope_id,session_id,lifetime_id);

-- Exact trusted relative paths/inode evidence are transient and purged after
-- success. No FK to sessions or artifacts may erase recovery ownership early.
CREATE TABLE neomd_service.session_deletion_inventory (
    operation_id uuid NOT NULL REFERENCES neomd_service.session_deletions(operation_id),
    ordinal integer NOT NULL CHECK(ordinal>=0),
    resource_kind text NOT NULL CHECK(resource_kind IN ('session','analysis')),
    resource_id uuid NOT NULL,
    session_id uuid,
    relative_path text NOT NULL,
    kind text NOT NULL CHECK(kind IN ('file','directory','object_alias')),
    device numeric(20,0) NOT NULL CHECK(device>=0),
    inode numeric(20,0) NOT NULL CHECK(inode>=0),
    size_bytes bigint NOT NULL CHECK(size_bytes>=0),
    mtime_ns bigint NOT NULL,
    ctime_ns bigint NOT NULL,
    link_count bigint NOT NULL CHECK(link_count>=1),
    removed boolean NOT NULL DEFAULT false,
    CHECK((resource_kind='session' AND session_id IS NOT NULL AND session_id=resource_id)
        OR (resource_kind='analysis' AND session_id IS NULL)),
    PRIMARY KEY(operation_id,ordinal),
    UNIQUE(operation_id,relative_path)
);

CREATE INDEX session_continuation_source ON neomd_service.sessions(scope_id,
    (snapshot #>> '{continuation,source_session_id}'))
    WHERE snapshot->'continuation' IS NOT NULL AND snapshot->'continuation'<>'null'::jsonb;
CREATE INDEX comparison_analysis_member_lookup ON neomd_service.comparison_analyses USING gin(request jsonb_path_ops);
INSERT INTO neomd_service.migrations(version) VALUES(16);
