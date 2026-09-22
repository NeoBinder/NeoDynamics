-- Preparation output is reserved before admission, shared with uploads/sessions.
ALTER TABLE neomd_service.preparation_operations
    ADD COLUMN IF NOT EXISTS reserved_output_bytes bigint NOT NULL DEFAULT 0
    CHECK (reserved_output_bytes >= 0);
-- Legacy pending writers have no trustworthy bound. Fence capacity until the
-- host can establish closure; lease loss alone cannot reclaim their files.
UPDATE neomd_service.preparation_operations
SET reserved_output_bytes=9007199254740991
WHERE state IN ('queued','running') AND reserved_output_bytes=0;
INSERT INTO neomd_service.migrations(version) VALUES (11);
