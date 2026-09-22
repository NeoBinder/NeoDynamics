"""Public coordinator lifecycle with real spawned, deliberately stuck uploaders."""

import asyncio
import multiprocessing
import os
import struct
import time
from dataclasses import replace

import pytest
from test_publication_runtime import build, materialized_fixture, settings, settle

from neomd_server import publication_runtime

pytestmark = pytest.mark.server_db


def stuck_uploader(spec, channel, send_early):
    if send_early == 'partial':
        os.write(channel.fileno(), struct.pack('!i', 1024) + b'unfinished')
    elif send_early:
        channel.send_bytes(b'not-a-completed-exchange')
    time.sleep(60)


class StuckContext:
    def __init__(self, send_early):
        self.context = multiprocessing.get_context('spawn')
        self.send_early = send_early
        self.processes = []

    def Pipe(self, **kwargs):
        return self.context.Pipe(**kwargs)

    def Process(self, *, target, args):
        process = self.context.Process(target=stuck_uploader, args=(*args, self.send_early))
        self.processes.append(process)
        return process

    def cleanup(self):
        for process in self.processes:
            if process.is_alive():
                process.kill()
            process.join(5)


@pytest.mark.parametrize('send_early', [False, True, 'partial'])
def test_deadline_kills_and_joins_child_even_after_payload(database_dsn, tmp_path, monkeypatch, send_early):
    monkeypatch.setattr(publication_runtime, '_JOIN_GRACE_SECONDS', 0.1)

    async def scenario():
        configured = replace(settings(), connect_timeout_seconds=1,
                             http_timeout_seconds=1, operation_timeout_seconds=2)
        ledger, _, coordinator = await build(database_dsn, tmp_path, None, settings_value=configured)
        context = StuckContext(send_early)
        coordinator.context = context
        try:
            scope, _, finalization, _ = await materialized_fixture(ledger, tmp_path / 'storage')
            await coordinator.start()

            async def child_joined():
                return bool(context.processes) and context.processes[0].exitcode is not None

            assert await settle(child_joined, seconds=4)
            assert context.processes[0].exitcode < 0
            record = await coordinator.journal.get(scope, finalization)
            assert record.receipt_bytes is None
            assert (tmp_path / 'storage' / 'objects').exists()
        finally:
            await coordinator.close()
            context.cleanup()
            await ledger.close()

    asyncio.run(scenario())


def test_shutdown_waits_for_cancelled_child_exit(database_dsn, tmp_path):
    async def scenario():
        ledger, _, coordinator = await build(database_dsn, tmp_path, None)
        coordinator.shutdown_timeout_seconds = 0.05
        context = StuckContext(False)
        coordinator.context = context
        try:
            scope, _, finalization, _ = await materialized_fixture(ledger, tmp_path / 'storage')
            await coordinator.start()

            async def child_started():
                return bool(context.processes) and context.processes[0].is_alive()

            assert await settle(child_started)
            await coordinator.close()
            assert all(process.exitcode is not None for process in context.processes)
            assert all(not process.is_alive() for process in context.processes)
            record = await coordinator.journal.get(scope, finalization)
            assert record.receipt_bytes is None
            operations = await coordinator.journal.operations(scope, finalization)
            assert operations.operations[0].state == 'possible'
            ref = (await coordinator.journal.jobs()).entries[0]
            assert ref.owner_closed and not ref.active_reserved
            assert ref.pending_reserved
        finally:
            await coordinator.close()
            context.cleanup()
            await ledger.close()

    asyncio.run(scenario())
