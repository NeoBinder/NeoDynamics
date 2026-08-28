from abc import ABC, abstractmethod


class BaseEngine(ABC):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def name(self):
        return ""

    @abstractmethod
    def get_positions(self):
        pass

    @abstractmethod
    def get_energy_forces(self):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abstractmethod
    def minimize_energy(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_last(self, *args, **kwargs):
        pass

    @abstractmethod
    def run_md(self, steps):
        pass
