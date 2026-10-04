-- Independent request tombstones for explicitly selected derived records.
CREATE TABLE neomd_service.analysis_retirements (
    scope_id uuid NOT NULL,
    analysis_operation_id uuid NOT NULL,
    analysis_request_id uuid NOT NULL,
    deletion_operation_id uuid NOT NULL,
    state text NOT NULL CHECK(state IN ('deleting','deleted')),
    PRIMARY KEY(scope_id,analysis_operation_id),
    UNIQUE(scope_id,analysis_request_id),
    FOREIGN KEY(scope_id,deletion_operation_id)
        REFERENCES neomd_service.session_deletions(scope_id,operation_id)
);
CREATE INDEX artifact_local_reference_owner ON neomd_service.artifacts(local_ref);
CREATE UNIQUE INDEX session_retirement_global_identity ON neomd_service.session_retirements(session_id);

-- Intent precedes unlink. An absent entry after restart is accepted only after
-- this exact owned mutation was recorded; replacement entries remain blocked.
ALTER TABLE neomd_service.session_deletion_inventory
    ADD COLUMN unlink_started boolean NOT NULL DEFAULT false,
    ADD COLUMN content_sha256 text CHECK(content_sha256 ~ '^[0-9a-f]{64}$');

-- The public scientific revision includes ordinary observations. The private
-- consent revision changes for every scientific/authority mutation but permits
-- observations and running-time samples to advance between preview and stop.
ALTER TABLE neomd_service.sessions ADD COLUMN deletion_revision bigint NOT NULL DEFAULT 1;
CREATE FUNCTION neomd_service.deletion_semantics(value jsonb) RETURNS jsonb
LANGUAGE sql IMMUTABLE AS $$ SELECT (value - ARRAY['service_revision','updated_at','observation','runtime_budget']) || jsonb_build_object('runtime_authority', jsonb_build_object('limit_ms',value #> '{runtime_budget,limit_ms}','stop',value #> '{runtime_budget,stop}')) $$;
CREATE FUNCTION neomd_service.advance_deletion_revision() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF neomd_service.deletion_semantics(NEW.snapshot) IS DISTINCT FROM neomd_service.deletion_semantics(OLD.snapshot)
       OR (NEW.bundle_id,NEW.protocol_id,NEW.worker_id) IS DISTINCT FROM (OLD.bundle_id,OLD.protocol_id,OLD.worker_id) THEN
        NEW.deletion_revision := OLD.deletion_revision + 1;
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER session_deletion_semantics BEFORE UPDATE ON neomd_service.sessions
    FOR EACH ROW EXECUTE FUNCTION neomd_service.advance_deletion_revision();
CREATE FUNCTION neomd_service.advance_deletion_child_revision() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    UPDATE neomd_service.sessions SET deletion_revision=deletion_revision+1
        WHERE session_id=COALESCE(NEW.session_id,OLD.session_id);
    RETURN COALESCE(NEW,OLD);
END $$;
CREATE TRIGGER deletion_command_revision AFTER INSERT OR UPDATE ON neomd_service.commands
    FOR EACH ROW EXECUTE FUNCTION neomd_service.advance_deletion_child_revision();
-- Publication ownership/fences are semantic despite not being exposed as
-- scientific snapshot fields. Give a fresh preview a distinct ordinary revision
-- and preserve the same correctly shaped durable snapshot event stream.
CREATE FUNCTION neomd_service.advance_deletion_binding_revision() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE snap jsonb; now_at timestamptz;
BEGIN
    IF TG_OP='UPDATE' AND (to_jsonb(NEW)-'lease_expires_at') IS NOT DISTINCT FROM (to_jsonb(OLD)-'lease_expires_at') THEN
        RETURN NEW;
    END IF;
    now_at := clock_timestamp();
    SELECT snapshot INTO snap FROM neomd_service.sessions WHERE session_id=NEW.session_id FOR UPDATE;
    IF snap IS NULL THEN RETURN NEW; END IF;
    snap := jsonb_set(snap,'{service_revision}',to_jsonb((snap->>'service_revision')::bigint+1));
    IF snap->'observation' IS NOT NULL AND snap->'observation'<>'null'::jsonb THEN
        snap := jsonb_set(snap,'{observation,freshness}',to_jsonb(CASE WHEN (snap #>> '{observation,fresh_until}')::timestamptz>now_at THEN 'fresh'::text ELSE 'stale'::text END));
    END IF;
    snap := jsonb_set(snap,'{updated_at}',to_jsonb(to_char(now_at AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS.US"Z"')));
    UPDATE neomd_service.sessions SET deletion_revision=deletion_revision+1,snapshot=snap WHERE session_id=NEW.session_id;
    INSERT INTO neomd_service.events(scope_id,session_id,event)
        SELECT scope_id,session_id,jsonb_build_object('session_id',session_id::text,
            'service_revision',(snap->>'service_revision')::bigint,'type','state_changed',
            'created_at',snap->>'updated_at','snapshot',snap,'command',NULL)
        FROM neomd_service.sessions WHERE session_id=NEW.session_id;
    RETURN NEW;
END $$;
CREATE TRIGGER deletion_publication_revision AFTER INSERT OR UPDATE ON neomd_service.publication_jobs
    FOR EACH ROW EXECUTE FUNCTION neomd_service.advance_deletion_binding_revision();
CREATE TRIGGER deletion_artifact_revision AFTER INSERT OR UPDATE ON neomd_service.artifacts
    FOR EACH ROW EXECUTE FUNCTION neomd_service.advance_deletion_binding_revision();
CREATE TABLE neomd_service.session_deletion_previews (
    scope_id uuid NOT NULL,
    preview_sha256 text NOT NULL,
    preview jsonb NOT NULL,
    semantic_revisions jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(scope_id,preview_sha256)
);
INSERT INTO neomd_service.migrations(version) VALUES(17);
