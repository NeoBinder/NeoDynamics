"""Remove only the new journal from an isolated older-schema test fixture."""


async def remove_publication_journal(connection):
    await remove_checkpoint_evidence(connection)
    await connection.execute('''DROP TABLE neomd_service.publication_operations,
        neomd_service.publication_uploads,neomd_service.publication_jobs,
        neomd_service.publication_destinations''')
    await connection.execute('DELETE FROM neomd_service.migrations WHERE version=8')


async def remove_checkpoint_evidence(connection):
    await connection.execute('DROP TABLE neomd_service.checkpoint_evidence_artifacts,neomd_service.checkpoint_evidence')
    await connection.execute('DELETE FROM neomd_service.migrations WHERE version=9')
