-- No inferred finalization state/backfill for historical engine attempts.
CREATE TABLE neomd_service.finalizations (
    finalization_id uuid PRIMARY KEY,
    scope_id uuid NOT NULL, session_id uuid NOT NULL UNIQUE,
    attempt_id uuid NOT NULL, generation bigint NOT NULL, epoch bigint NOT NULL,
    seal jsonb, deadline_seconds integer NOT NULL CHECK (deadline_seconds > 0),
    status text NOT NULL CHECK (status IN ('pending','running','complete','failed','interrupted')),
    owner_id uuid NOT NULL, lease_id uuid, lease_expires_at timestamptz,
    progress_sequence bigint NOT NULL DEFAULT 0 CHECK (progress_sequence >= 0),
    deadline_at timestamptz, result_sha256 text, artifact_ids uuid[],
    FOREIGN KEY(scope_id,session_id) REFERENCES neomd_service.sessions(scope_id,session_id),
    CHECK (seal IS NOT NULL OR (status IN ('failed','interrupted')
        AND result_sha256 IS NULL AND artifact_ids IS NULL AND lease_id IS NULL))
);
INSERT INTO neomd_service.migrations(version) VALUES (4);
