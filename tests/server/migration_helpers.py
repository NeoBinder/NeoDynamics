"""Reproduce older schemas only in the explicitly disposable PG test fixture."""


async def remove_publication_journal(connection):
    await remove_checkpoint_evidence(connection)
    await connection.execute('''DROP TABLE neomd_service.publication_operations,
        neomd_service.publication_uploads,neomd_service.publication_jobs,
        neomd_service.publication_destinations''')
    await connection.execute('DELETE FROM neomd_service.migrations WHERE version>=8')


async def remove_checkpoint_evidence(connection):
    if not (await connection.fetchval('SELECT current_database()')).startswith('neomd_test_'):
        raise RuntimeError('schema downgrade requires the disposable test database')
    await connection.execute('''DROP TRIGGER IF EXISTS session_deletion_semantics ON neomd_service.sessions;
        DROP TRIGGER IF EXISTS deletion_command_revision ON neomd_service.commands;
        DROP TRIGGER IF EXISTS deletion_artifact_revision ON neomd_service.artifacts;
        DROP TRIGGER IF EXISTS deletion_publication_revision ON neomd_service.publication_jobs;
        DROP FUNCTION IF EXISTS neomd_service.advance_deletion_revision(),
            neomd_service.advance_deletion_child_revision(),
            neomd_service.advance_deletion_binding_revision(),
            neomd_service.deletion_semantics(jsonb);
        DROP TABLE IF EXISTS neomd_service.analysis_retirements,
            neomd_service.session_deletion_previews,neomd_service.session_deletion_inventory,
            neomd_service.session_lifetime_members,neomd_service.session_lifetimes,
            neomd_service.session_retirements,neomd_service.session_deletions,
            neomd_service.comparison_analyses;
        DROP INDEX IF EXISTS neomd_service.session_continuation_source,
            neomd_service.artifact_local_reference_owner;
        ALTER TABLE neomd_service.sessions DROP COLUMN IF EXISTS deletion_revision;
        ALTER TABLE neomd_service.publication_jobs DROP COLUMN IF EXISTS cleanup_after,
            DROP COLUMN IF EXISTS cleanup_done''')
    await connection.execute('DROP TABLE neomd_service.checkpoint_evidence_artifacts,neomd_service.checkpoint_evidence')
    await connection.execute('DELETE FROM neomd_service.migrations WHERE version>=9')
