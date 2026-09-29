-- Published Min/Eq outputs remain locally available for a bounded continuation
-- window. Cleanup completion and capacity release are durable and resumable.
ALTER TABLE neomd_service.publication_jobs
    ADD COLUMN cleanup_after timestamptz,
    ADD COLUMN cleanup_done boolean NOT NULL DEFAULT false;

-- Earlier completed jobs may already have been deleted. The guarded sweeper
-- reconciles them without granting old Min/Eq sessions a new 24-hour window.
UPDATE neomd_service.publication_jobs
SET cleanup_after = clock_timestamp()
WHERE state = 'complete';

CREATE INDEX publication_cleanup_due
    ON neomd_service.publication_jobs(pool_id, cleanup_after, journal_id)
    WHERE state = 'complete' AND NOT cleanup_done;

INSERT INTO neomd_service.migrations(version) VALUES (15);
