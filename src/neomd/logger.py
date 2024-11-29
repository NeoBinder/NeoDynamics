import collections
import logging


def get_logger(logger_name, fpath=None, level=logging.DEBUG):
    """get_logger.

    Parameters
    ----------
    fpath :
        fpath logging path
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    if not logger.hasHandlers():
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.DEBUG)
        logger.addHandler(stream_handler)
    if fpath is not None:
        file_handler = logging.FileHandler(fpath)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    return logger


class SimulationLogger:

    def __init__(self, simulation):
        self.simulation = simulation
        self.integrator = simulation.integrator
        self.logger = get_logger("simulationlogger", "./output")

    @staticmethod
    def logging_list(logging_with_level, lst):
        for _l in lst:
            logging_with_level(_l)

    @staticmethod
    def logging_dict(logging_with_level, res_dict):
        for k, v in collections.OrderedDict(sorted(res_dict.items())).items():
            logging_with_level("{} : {}".format(k, v))

    def logging_algorithms(self):
        integrator = self.simulation.integrator
        res = []
        for i in range(integrator.getNumComputations()):
            res.append(integrator.getComputationStep(i))
        self.logger.info("-------------algorithms-------------")
        self.logging_list(self.logger.info, res)

    def logging_integrator_variables(self):
        integrator = self.simulation.integrator
        res = {}
        for index in range(0, integrator.getNumGlobalVariables()):
            name = integrator.getGlobalVariableName(index)
            res[name] = integrator.getGlobalVariableByName(name)
        self.logger.info("-------------variables-------------")
        self.logging_dict(self.logger.info, res)
