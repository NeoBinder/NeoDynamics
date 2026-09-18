"""Public owner clock and service-driver accounting, without wall-clock sleeps."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from test_interactive import config

from neomd.kernel.fake import FakeKernel
from neomd.sinks import MemorySink
from neomd_server.budget import RunningBudget
from neomd_server.driver import open_session


class Clock:
    def __init__(self):
        self.now = 0

    def __call__(self):
        return self.now

    def advance(self, ns):
        self.now += ns


@pytest.mark.parametrize('method', ['md', 'rbfe', 'gamd', 'min'])
def test_driver_excludes_preparation_pause_and_finalization_across_windows(method):
    clock = Clock()
    budget = RunningBudget(1000, clock=clock)

    class TimedKernel(FakeKernel):
        def __init__(self, spec):
            clock.advance(1_000_000)  # preparation must never consume the limit
            super().__init__(spec)

        def step(self, count):
            clock.advance(count * 10)
            super().step(count)

        def minimize(self, *args, **kwargs):
            clock.advance(kwargs['max_iterations'] * 10)
            return super().minimize(*args, **kwargs)

        def snapshot(self):
            # At terminal/window finalization, this slow checkpoint is excluded.
            clock.advance(10_000)
            return super().snapshot()

    run = open_session(config(method, steps=6), kernel_factory=TimedKernel,
                       sink=MemorySink(), running_budget=budget)
    assert budget.elapsed_ns == 0
    run.advance(2)
    assert budget.elapsed_ns == 20
    run.command({'type': 'pause'})
    clock.advance(1_000_000)
    before = run.status()['step']
    assert run.advance(2)['step'] == before and budget.elapsed_ns == 20
    run.command({'type': 'resume'})
    run.advance(2)
    assert budget.elapsed_ns == 40
    while run.status()['state'] != 'completed':
        run.advance(2)
    expected = {'md': 60, 'rbfe': 120, 'min': 120, 'gamd': 200}[method]
    assert budget.elapsed_ns == expected
    clock.advance(1_000_000)
    run.close()
    assert budget.elapsed_ns == expected


def test_monotonic_budget_samples_preserve_fraction_and_owner_identity():
    clock = Clock()
    budget = RunningBudget(1_000_000, clock=clock)
    budget.running(True)
    clock.advance(999_999)
    assert not budget.exhausted
    budget.running(False)
    clock.advance(8_000_000)
    budget.running(True)
    clock.advance(1)
    assert budget.exhausted
    first, second = budget.sample(), budget.sample()
    assert first['elapsed_ns'] == second['elapsed_ns'] == 1_000_000
    assert (first['sequence'], second['sequence']) == (1, 2)
    with ThreadPoolExecutor(1) as executor:
        with pytest.raises(RuntimeError, match='owner thread'):
            executor.submit(budget.sample).result()
