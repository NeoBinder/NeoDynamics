"""Read-only completion polling for explicit synthetic input-operation requests."""

import asyncio
from uuid import UUID, uuid5

from neomd_server.ledger import LedgerError


async def completed(inputs, scope, operation):
    async with asyncio.timeout(inputs.verification_timeout_seconds + 5):
        while operation['state'] in {'accepted', 'running'}:
            await asyncio.sleep(.025)
            operation = await inputs.input_operation(scope, operation['operation_id'])
    if operation['state'] != 'succeeded':
        raise LedgerError(operation['error']['code'])
    return operation['result']


async def commit_bundle(inputs, scope, upload_id):
    await completed(inputs, scope, await inputs.commit_bundle(
        scope, upload_id, {'request_id': str(uuid5(UUID(upload_id), 'fixture-prepared-commit'))},
    ))
    return await inputs.bundle_upload(scope, upload_id)


async def commit_coordinates(inputs, scope, upload_id):
    await completed(inputs, scope, await inputs.commit_coordinates(
        scope, upload_id, {'request_id': str(uuid5(UUID(upload_id), 'fixture-coordinate-commit'))},
    ))
    return await inputs.coordinate_upload(scope, upload_id)


async def register_protocol(inputs, scope, request):
    return await completed(inputs, scope, await inputs.register_protocol(scope, request))
