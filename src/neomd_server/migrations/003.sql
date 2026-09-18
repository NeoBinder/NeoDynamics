-- Run with service processes stopped. Historical attempted elapsed time is unknowable.
CREATE TABLE neomd_service.runtime_policy (
    pool_id text PRIMARY KEY REFERENCES neomd_service.admission(pool_id),
    max_running_seconds integer NOT NULL CHECK (max_running_seconds > 0)
);
ALTER TABLE neomd_service.sessions
    ADD COLUMN runtime_sample_sequence bigint NOT NULL DEFAULT 0 CHECK (runtime_sample_sequence >= 0),
    ADD COLUMN runtime_elapsed_ns bigint CHECK (runtime_elapsed_ns >= 0);

-- Preserve old event/receipt history. Append explicit migration observations;
-- never estimate runtime from wall age or release an unconfirmed allocation.
DO $$
DECLARE r record; c record; snap jsonb; new_receipt jsonb; stamp text; code text;
BEGIN
  stamp := to_char(clock_timestamp() AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"');
  FOR r IN SELECT * FROM neomd_service.sessions FOR UPDATE LOOP
    snap := r.snapshot || jsonb_build_object('runtime_budget', jsonb_build_object(
      'limit_ms', NULL, 'elapsed_ms', CASE WHEN r.snapshot->>'attempt_id' IS NULL THEN 0 ELSE NULL END,
      'accounting', CASE WHEN r.snapshot->>'attempt_id' IS NULL THEN 'not_started' ELSE 'unknown' END,
      'stop', NULL));
    IF r.snapshot->>'attempt_id' IS NULL THEN
      UPDATE neomd_service.sessions SET runtime_elapsed_ns=0 WHERE session_id=r.session_id;
    ELSIF r.snapshot->>'state' NOT IN ('completed','stopped','failed','expired') THEN
      FOR c IN SELECT * FROM neomd_service.commands WHERE session_id=r.session_id
          AND status IN ('accepted','claimed') FOR UPDATE LOOP
        code := CASE WHEN c.status='claimed' THEN 'outcome_indeterminate' ELSE 'worker_lost' END;
        snap := snap || jsonb_build_object('service_revision',(snap->>'service_revision')::bigint+1,'updated_at',stamp);
        new_receipt := c.receipt || jsonb_build_object(
          'status',CASE WHEN c.status='claimed' THEN 'indeterminate' ELSE 'rejected' END,
          'resolved_at',stamp,'error',jsonb_build_object('code',code,'message',code,'retryable',false),
          'service_revision',snap->'service_revision');
        UPDATE neomd_service.commands SET status=new_receipt->>'status',receipt=new_receipt
          WHERE session_id=c.session_id AND command_id=c.command_id;
        INSERT INTO neomd_service.events(scope_id,session_id,event) VALUES(r.scope_id,r.session_id,
          jsonb_build_object('session_id',r.session_id,'service_revision',snap->'service_revision',
            'type','command_resolved','created_at',stamp,'snapshot',snap,'command',new_receipt));
      END LOOP;
      snap := snap || jsonb_build_object('state','recovery_required','controller',NULL,
        'generation',(snap->>'generation')::bigint+1);
      UPDATE neomd_service.sessions SET worker_lease_expires_at=clock_timestamp()
        WHERE session_id=r.session_id;
    END IF;
    snap := snap || jsonb_build_object('service_revision',(snap->>'service_revision')::bigint+1,'updated_at',stamp);
    UPDATE neomd_service.sessions SET snapshot=snap WHERE session_id=r.session_id;
    INSERT INTO neomd_service.events(scope_id,session_id,event) VALUES(r.scope_id,r.session_id,
      jsonb_build_object('session_id',r.session_id,'service_revision',snap->'service_revision',
        'type','state_changed','created_at',stamp,'snapshot',snap,'command',NULL));
  END LOOP;
END $$;
INSERT INTO neomd_service.migrations(version) VALUES (3);
