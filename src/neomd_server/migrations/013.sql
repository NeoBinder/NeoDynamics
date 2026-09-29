-- Component review, package production/import, and assembly share the fenced
-- preparation lane. They bind exact stored input artifacts instead of one
-- mandatory canonical source upload.
ALTER TABLE neomd_service.preparation_operations
    ALTER COLUMN source_upload_id DROP NOT NULL;
ALTER TABLE neomd_service.preparation_operations
    DROP CONSTRAINT IF EXISTS preparation_operations_kind_check;
ALTER TABLE neomd_service.preparation_operations
    ADD CONSTRAINT preparation_operations_kind_check
    CHECK (kind IN ('protein_preparation', 'topology_preparation',
                   'component_review', 'component_generate',
                   'component_import', 'component_assemble'));
-- Component idempotency is per owner and request, across all four kinds.
CREATE UNIQUE INDEX IF NOT EXISTS preparation_component_request_scope
    ON neomd_service.preparation_operations(scope_id, request_id)
    WHERE kind IN ('component_review', 'component_generate',
                   'component_import', 'component_assemble');
ALTER TABLE neomd_service.preparation_artifacts
    DROP CONSTRAINT IF EXISTS preparation_artifacts_role_check;
ALTER TABLE neomd_service.preparation_artifacts
    ADD CONSTRAINT preparation_artifacts_role_check
    CHECK (role IN ('manifest','topology','system','coordinates','mapping',
                   'report','source','forcefield','ligand','state','instance',
                   'package','assembly','qc','prepared','asset','log'));
INSERT INTO neomd_service.migrations(version) VALUES (13);
