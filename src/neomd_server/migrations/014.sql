CREATE TABLE neomd_service.comparison_analyses (
    operation_id uuid PRIMARY KEY,
    scope_id uuid NOT NULL,
    request_id uuid NOT NULL,
    comparison_id uuid NOT NULL,
    request_sha256 text NOT NULL,
    request jsonb NOT NULL,
    state text NOT NULL CHECK (state IN ('queued','running','succeeded','failed','interrupted')),
    receipt jsonb,
    error jsonb,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE(scope_id,request_id)
);
CREATE INDEX comparison_analyses_scope_comparison
    ON neomd_service.comparison_analyses(scope_id,comparison_id,created_at);
INSERT INTO neomd_service.migrations(version) VALUES (14);
