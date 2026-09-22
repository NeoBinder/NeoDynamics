-- Exact new producer evidence only. Legacy checkpoint identities stay unproved.
CREATE TABLE neomd_service.checkpoint_evidence (
    checkpoint_id uuid PRIMARY KEY,
    scope_id uuid NOT NULL, session_id uuid NOT NULL,
    attempt_id uuid NOT NULL, generation bigint NOT NULL CHECK(generation > 0),
    epoch bigint NOT NULL CHECK(epoch >= 0), step bigint NOT NULL CHECK(step >= 0),
    control_revision bigint NOT NULL CHECK(control_revision >= 0),
    cause_kind text NOT NULL CHECK(cause_kind IN ('command','budget_stop','terminal')),
    cause_id uuid NOT NULL,
    metadata_artifact_id uuid NOT NULL UNIQUE REFERENCES neomd_service.artifacts(artifact_id),
    metadata_bytes bytea NOT NULL CHECK(octet_length(metadata_bytes) BETWEEN 1 AND 1048576),
    inventory_sha256 text NOT NULL CHECK(inventory_sha256 ~ '^[0-9a-f]{64}$'),
    artifact_count integer NOT NULL CHECK(artifact_count > 1),
    UNIQUE(scope_id,session_id,cause_kind,cause_id),
    FOREIGN KEY(scope_id,session_id) REFERENCES neomd_service.sessions(scope_id,session_id)
);
CREATE INDEX checkpoint_attempt ON neomd_service.checkpoint_evidence(scope_id,session_id,attempt_id,checkpoint_id);
CREATE TABLE neomd_service.checkpoint_evidence_artifacts (
    checkpoint_id uuid NOT NULL REFERENCES neomd_service.checkpoint_evidence(checkpoint_id),
    ordinal integer NOT NULL CHECK(ordinal >= 0),
    artifact_id uuid NOT NULL UNIQUE REFERENCES neomd_service.artifacts(artifact_id),
    local_ref uuid NOT NULL CHECK(local_ref = artifact_id),
    relative_path text NOT NULL,
    descriptor jsonb NOT NULL CHECK(jsonb_typeof(descriptor) = 'object'),
    stat jsonb NOT NULL CHECK(jsonb_typeof(stat) = 'array' AND jsonb_array_length(stat) = 5),
    PRIMARY KEY(checkpoint_id,ordinal), UNIQUE(checkpoint_id,relative_path)
);
INSERT INTO neomd_service.migrations(version) VALUES(9);
