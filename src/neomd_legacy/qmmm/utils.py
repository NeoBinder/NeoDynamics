import os
import numpy as np
from openmm import unit, app


def openmm2pint_unit(value):
    value_dimless = value.magnitude
    unit_str = str(value.units)
    unit_ls = unit_str.split()
    value_unit = getattr(unit, unit_ls[0])
    for i in range(int((len(unit_ls) - 1) / 2)):
        oper = unit_ls[2 * i + 1]
        _unit = unit_ls[2 * i + 2]
        tmp_unit = getattr(unit, _unit)
        if oper == "/":
            value_unit = value_unit / tmp_unit
        elif oper == "*":
            value_unit = value_unit * tmp_unit
        else:
            raise AssertionError(
                "unknown operater '{}' in unit,should be * or / \n".format(oper)
            )
    return unit.Quantity(value_dimless) * value_unit


# def pint2openmm_unit(in_value, units):
#     unit_name = in_value.unit.get_name()
#     out_value = in_value.value_in_unit(in_value.unit) * units(unit_name)
#     return out_value


class ScipyMinimizeCallBackSave:

    def __init__(
        self, qmmm_handler, basedir, logger, save_last=True, increment_constraints=True
    ):
        self.logger = logger
        self.count = 0
        self.md_wrapper = qmmm_handler.md_wrapper
        self.topology = self.md_wrapper.topology
        self.basedir = basedir
        self.last_xk = self.md_wrapper.positions.value_in_unit(unit.nanometer).reshape(
            -1
        )
        self.save_last = save_last
        self.increment_constraints = increment_constraints

    def save(self, xk):
        diff = np.absolute(xk - self.last_xk)
        self.logger.info("max pos difference:{}".format(max(diff)))
        if self.save_last:
            self.last_xk = xk
            xk = xk.reshape(xk.shape[0] // 3, 3)
            outfname = os.path.join(self.basedir, "pos{}.pdb".format(self.count))
            app.PDBFile.writeFile(
                self.topology, unit.Quantity(xk) * unit.nanometer, open(outfname, "w")
            )
        if self.increment_constraints:
            self.md_wrapper.increment_constraints(increment_ratio=1.5)
            self.logger.info(
                "current constraints k:{}".format(self.md_wrapper.constraints_k)
            )
        self.count += 1
