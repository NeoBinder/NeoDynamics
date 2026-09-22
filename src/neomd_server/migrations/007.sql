-- Immutable completion evidence only; historical complete rows are not backfilled.
CREATE TABLE neomd_service.finalization_evidence (
    finalization_id uuid PRIMARY KEY REFERENCES neomd_service.finalizations(finalization_id),
    result_artifact_id uuid NOT NULL,
    inventory_sha256 text NOT NULL CHECK (inventory_sha256 ~ '^[0-9a-f]{64}$'),
    inventory_size_bytes integer NOT NULL CHECK (inventory_size_bytes BETWEEN 1 AND 8388608),
    artifact_count integer NOT NULL CHECK (artifact_count BETWEEN 1 AND 4096),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE neomd_service.finalization_evidence_artifacts (
    finalization_id uuid NOT NULL REFERENCES neomd_service.finalization_evidence(finalization_id),
    ordinal integer NOT NULL CHECK (ordinal BETWEEN 0 AND 4095),
    artifact_id uuid NOT NULL, local_ref uuid NOT NULL CHECK (local_ref = artifact_id),
    descriptor jsonb NOT NULL CHECK (jsonb_typeof(descriptor) = 'object'),
    stat jsonb NOT NULL CHECK (jsonb_typeof(stat) = 'array' AND jsonb_array_length(stat) = 5),
    PRIMARY KEY(finalization_id, ordinal),
    UNIQUE(finalization_id, artifact_id)
);
INSERT INTO neomd_service.migrations(version) VALUES (7);
