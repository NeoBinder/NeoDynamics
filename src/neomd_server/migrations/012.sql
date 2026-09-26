-- The same owner-scoped preparation ledger admits canonical topology sources.
ALTER TABLE neomd_service.preparation_operations
    DROP CONSTRAINT IF EXISTS preparation_operations_kind_check;
ALTER TABLE neomd_service.preparation_operations
    ADD CONSTRAINT preparation_operations_kind_check
    CHECK (kind IN ('protein_preparation', 'topology_preparation'));
ALTER TABLE neomd_service.preparation_artifacts
    DROP CONSTRAINT IF EXISTS preparation_artifacts_role_check;
ALTER TABLE neomd_service.preparation_artifacts
    ADD CONSTRAINT preparation_artifacts_role_check
    CHECK (role IN ('manifest','topology','system','coordinates','mapping','report',
                   'source','forcefield','log','ligand'));
INSERT INTO neomd_service.migrations(version) VALUES (12);
