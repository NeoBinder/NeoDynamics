import logging
import os


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
        fpath = os.path.abspath(fpath)
        if not any(
            isinstance(handler, logging.FileHandler)
            and handler.baseFilename == fpath
            for handler in logger.handlers
        ):
            file_handler = logging.FileHandler(fpath)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
    return logger
