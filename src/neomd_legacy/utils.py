import os

import openmm


def get_platform(method="cuda", cuda_index="0"):
    """
    Get the OpenMM platform configuration.

    This function returns the OpenMM platform configuration based on the specified method and CUDA index.

    Parameters
    ----------
    method : str, optional
        The method to use for the platform. Default is "cuda".
    cuda_index : str, optional
        The CUDA index to use. Default is "0".

    Returns
    -------
    dict
        The OpenMM platform configuration.

    Raises
    ------
    NotImplementedError
        If the method is not "cuda" or "cpu".

    """
    config = {}
    if method.lower() == "cuda":
        platform = openmm.Platform.getPlatformByName("CUDA")
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        device_index = visible_devices.split(",")[0] if visible_devices else cuda_index
        platform_properties = {
            "CudaPrecision": "single",
            "DeviceIndex": device_index,
        }
        config["platform"] = platform
        config["platformProperties"] = platform_properties
    elif method.lower() == "cpu":
        platform = openmm.Platform.getPlatformByName("CPU")
        config["platform"] = platform
    else:
        raise NotImplementedError(
            'platform method "{}" is not supported, use "cuda" or "cpu"'.format(method)
        )

    return config


def idstr2list(idstr):
    """
    Convert a string of integers into a list of integers.

    Parameters
    ----------
    idstr : str
        A string of integers separated by commas.

    Returns
    -------
    list
        A list of integers.

    """
    assert isinstance(idstr, str)
    return list(map(int, idstr.split(",")))


def floatstr2list(in_str):
    """
    Convert a string containing floating-point numbers into a list of floats.

    Parameters:
    in_str (str): A string containing floating-point numbers separated by commas.

    Returns:
    list: A list of floating-point numbers.

    Assertion:
    The input must be of string type.
    """
    assert isinstance(in_str, str)
    return list(map(float, in_str.split(",")))


def check_config(config):
    allow_set = {
        "method",
        "temperature",
        "barostat",
        "seed",
        "integrator",
        "continue_md",
        "colvars",
        "restraint",
        "meta_set",
        "qmmm",
        "steps",
        "input_files",
        "output",
        "min_params",
        "debug",
        "system_modification",
    }
    for k in config.keys():
        if k not in allow_set:
            raise ValueError('config with key "{}" is not allow'.format(k))

