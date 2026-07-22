from openmm import unit
from openmm.unit import nanometer  # type: ignore
from openmm.openmm import CustomCentroidBondForce, CustomTorsionForce
from openmm.app.metadynamics import BiasVariable

from neomd.utils import idstr2list


def _make_bias_variable(force, config, cv_unit, default_periodic):
    suffix = "nm" if cv_unit is nanometer else "degree"
    return BiasVariable(
        force,
        getattr(config, "min_cv_" + suffix) * cv_unit,
        getattr(config, "max_cv_" + suffix) * cv_unit,
        getattr(config, "biasWidth_" + suffix) * cv_unit,
        config.get("is_period", default_periodic),
        gridWidth=config.bins,
    )


def generate_colvar_distance(config):
    cv_dis = CustomCentroidBondForce(2, "distance(g1,g2)")

    grp1_idx = idstr2list(config.grp1_idx)
    grp2_idx = idstr2list(config.grp2_idx)
    cv_dis.addGroup(grp1_idx)
    cv_dis.addGroup(grp2_idx)
    cv_dis.addBond([0, 1])
    cv_dis.setUsesPeriodicBoundaryConditions(True)
    return _make_bias_variable(cv_dis, config, nanometer, False)


def generate_colvar_distance_ref(config):
    cv_dis = CustomCentroidBondForce(
        1,
        "(dx^2 + dy^2 + dz^2)^0.5; \
                                        dx = x1 - x0; \
                                        dy = y1 - y0; \
                                        dz = z1 - z0",
    )
    cv_dis.addPerBondParameter("x0")
    cv_dis.addPerBondParameter("y0")
    cv_dis.addPerBondParameter("z0")

    cv_dis.addGroup(idstr2list(config.particles))
    config.ref_pos = [float(x) for x in config.ref_pos.split(",")]
    cv_dis.addBond([0], config.ref_pos * nanometer)
    cv_dis.setUsesPeriodicBoundaryConditions(True)
    return _make_bias_variable(cv_dis, config, nanometer, False)


def generate_colvar_min_distances(config):
    cv_dis = CustomCentroidBondForce(3, "min(distance(g1,g3),distance(g2,g3))")
    cv_dis.addGroup(idstr2list(config.min1_idx1))
    cv_dis.addGroup(idstr2list(config.min2_idx1))
    cv_dis.addGroup(idstr2list(config.min_idx2))
    cv_dis.addBond([0, 1, 2])
    cv_dis.setUsesPeriodicBoundaryConditions(True)
    return _make_bias_variable(cv_dis, config, nanometer, False)


def generate_colvar_dihedral(config):
    cv_dih = CustomTorsionForce("theta")
    cv_dih.addTorsion(
        idstr2list(config.grp1_idx)[0],
        idstr2list(config.grp2_idx)[0],
        idstr2list(config.grp3_idx)[0],
        idstr2list(config.grp4_idx)[0],
    )
    cv_dih.setUsesPeriodicBoundaryConditions(True)
    return _make_bias_variable(cv_dih, config, unit.degree, True)


def generate_colvar_angle(config):

    cv_ang = CustomCentroidBondForce(3, "angle(g1,g2,g3)")
    cv_ang.addGroup(idstr2list(config.grp1_idx))
    cv_ang.addGroup(idstr2list(config.grp2_idx))
    cv_ang.addGroup(idstr2list(config.grp3_idx))
    cv_ang.addBond([0, 1, 2])
    cv_ang.setUsesPeriodicBoundaryConditions(True)
    return _make_bias_variable(cv_ang, config, unit.degree, False)


_COLVAR_FUNCS = {
    "distance": generate_colvar_distance,
    "dihedral": generate_colvar_dihedral,
    "angle": generate_colvar_angle,
    "min_distances": generate_colvar_min_distances,
    "distance_ref": generate_colvar_distance_ref,
}


def generate_colvar(colvar_config):
    if colvar_config["type"] not in _COLVAR_FUNCS:
        raise NotImplementedError(
            "colvar type:{} not defined".format(colvar_config["type"])
        )
    colvar = _COLVAR_FUNCS[colvar_config["type"]](colvar_config)
    return colvar
