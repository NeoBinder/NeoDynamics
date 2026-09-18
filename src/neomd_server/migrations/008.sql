-- Local publication intent/ownership only: no remote or PVC cleanup authority.
CREATE TABLE neomd_service.publication_destinations (
    pool_id text PRIMARY KEY REFERENCES neomd_service.admission(pool_id),
    provider_id uuid NOT NULL, origin text NOT NULL,
    max_active integer NOT NULL CHECK(max_active > 0),
    max_pending integer NOT NULL CHECK(max_pending >= max_active)
);
CREATE TABLE neomd_service.publication_jobs (
    journal_id uuid PRIMARY KEY,
    scope_id uuid NOT NULL, session_id uuid NOT NULL,
    finalization_id uuid NOT NULL UNIQUE REFERENCES neomd_service.finalization_evidence(finalization_id),
    pool_id text NOT NULL REFERENCES neomd_service.publication_destinations(pool_id),
    provider_id uuid NOT NULL, origin text NOT NULL,
    request_id uuid NOT NULL, request_bytes bytea NOT NULL CHECK(octet_length(request_bytes) BETWEEN 1 AND 4194304),
    candidate_bytes bytea NOT NULL CHECK(octet_length(candidate_bytes) BETWEEN 1 AND 1048576),
    state text NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','complete','held')),
    pending_reserved boolean NOT NULL DEFAULT true, active_reserved boolean NOT NULL DEFAULT false,
    owner_id uuid, lease_id uuid, fence bigint NOT NULL DEFAULT 0 CHECK(fence BETWEEN 0 AND 9007199254740991),
    lease_expires_at timestamptz, owner_closed boolean NOT NULL DEFAULT true,
    query_required boolean NOT NULL DEFAULT false,
    exchange_id uuid, exchange_bytes bytea CHECK(octet_length(exchange_bytes) <= 4096),
    publication_id uuid, geometry_sha256 text, remote_state text,
    observed_publication_id uuid, observed_geometry_sha256 text, observed_receipt_sha256 text,
    receipt_id uuid, receipt_bytes bytea CHECK(octet_length(receipt_bytes) BETWEEN 1 AND 4194304),
    receipt_sha256 text,
    stale_evidence jsonb, conflict_evidence jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(scope_id,request_id),
    FOREIGN KEY(scope_id,session_id) REFERENCES neomd_service.sessions(scope_id,session_id),
    CHECK((receipt_bytes IS NULL) = (receipt_sha256 IS NULL)),
    CHECK(state <> 'complete' OR receipt_bytes IS NOT NULL),
    CHECK((exchange_id IS NULL) = (exchange_bytes IS NULL))
);
CREATE INDEX publication_jobs_pool ON neomd_service.publication_jobs(pool_id,journal_id);
CREATE TABLE neomd_service.publication_uploads (
    journal_id uuid NOT NULL REFERENCES neomd_service.publication_jobs(journal_id),
    upload_id uuid NOT NULL, artifact_id uuid NOT NULL,
    geometry_bytes bytea NOT NULL CHECK(octet_length(geometry_bytes) BETWEEN 1 AND 1024),
    part_count integer NOT NULL CHECK(part_count BETWEEN 0 AND 10000),
    expires_at timestamptz NOT NULL, state text NOT NULL,
    PRIMARY KEY(journal_id,upload_id), UNIQUE(journal_id,artifact_id)
);
CREATE TABLE neomd_service.publication_operations (
    journal_id uuid NOT NULL REFERENCES neomd_service.publication_jobs(journal_id),
    operation_id uuid NOT NULL,
    kind text NOT NULL CHECK(kind IN ('create','part','action')),
    upload_id uuid, part_number integer, action text,
    intent_bytes bytea NOT NULL CHECK(octet_length(intent_bytes) BETWEEN 1 AND 4194304),
    state text NOT NULL DEFAULT 'unsent' CHECK(state IN ('unsent','possible','accepted','held')),
    send_exchange_id uuid, send_grant_bytes bytea CHECK(octet_length(send_grant_bytes) <= 4096),
    outcome jsonb, terminal_sha256 text, conflict_evidence jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(journal_id,operation_id),
    CHECK((send_exchange_id IS NULL) = (send_grant_bytes IS NULL))
);
CREATE UNIQUE INDEX publication_one_create ON neomd_service.publication_operations(journal_id) WHERE kind='create';
CREATE UNIQUE INDEX publication_one_part ON neomd_service.publication_operations(journal_id,upload_id,part_number) WHERE kind='part';
CREATE UNIQUE INDEX publication_one_action ON neomd_service.publication_operations(journal_id,upload_id,action) WHERE kind='action';
INSERT INTO neomd_service.migrations(version) VALUES(8);
