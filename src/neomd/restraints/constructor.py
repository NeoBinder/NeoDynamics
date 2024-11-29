import numpy as np
from openmm import unit
from openmm.openmm import CustomCentroidBondForce

from neomd.utils import floatstr2list, idstr2list


def generate_restraint(restraint_config, system=None):
    if restraint_config["type"] == "sphere":
        restraint = generate_restraint_sphere(restraint_config)
    elif restraint_config["type"] == "funnel":
        restraint = generate_restraint_funnel(restraint_config)
    elif restraint_config["type"] == "distance":
        restraint = generate_restraint_distance(restraint_config)
    elif restraint_config["type"] == "angle":
        restraint = generate_restraint_angle(restraint_config)
    elif restraint_config["type"] == "dihedral":
        restraint = generate_restraint_dihedral(restraint_config)
    elif restraint_config["type"] == "ref_file":
        restraint = generate_restraint_ref_file(restraint_config, system=system)
    elif restraint_config["type"] == "dist_ref_position":
        restraint = generate_dist_ref_position(restraint_config)
    elif restraint_config["type"] == "xyz_box":
        restraint = generate_xyz_box(restraint_config)
    elif restraint_config["type"] == "vec_restraint":
        restraint = generate_vec_restraint(restraint_config)
    elif restraint_config["type"] == "test":
        restraint = force_test(restraint_config)
    else:
        raise NotImplementedError(
            "restraint type:{} not defined".format(restraint_config["type"])
        )
    return restraint


def generate_restraint_sphere(restraint_config):
    # restain in a sphere wall.
    _name = restraint_config.name
    restraint_config.restr_grp = idstr2list(restraint_config.restr_grp)
    restraint_config.cent_grp = idstr2list(restraint_config.cent_grp)
    k_r = restraint_config.restr_k * unit.kilojoules_per_mole
    radius = restraint_config["radius_nm"] * unit.nanometer
    sphere_r = CustomCentroidBondForce(
        2, "(k{0}/2)*(max(distance(g1,g2) - radius{0}, 0)^2)".format(_name)
    )
    sphere_r.addGroup(restraint_config.restr_grp)
    sphere_r.addGroup(restraint_config.cent_grp)
    sphere_r.addBond([0, 1])
    sphere_r.addGlobalParameter("k{}".format(_name), k_r)
    sphere_r.addGlobalParameter("radius{}".format(_name), radius)
    sphere_r.setUsesPeriodicBoundaryConditions(True)
    return sphere_r


def generate_vec_restraint(restraint_config):
    # restain in a sphere wall.
    _name = restraint_config.name
    restraint_config.vec_grp1 = idstr2list(restraint_config.vec_grp1)
    restraint_config.vec_grp2 = idstr2list(restraint_config.vec_grp2)
    restraint_config.pos_ref1_nm = (
        floatstr2list(restraint_config.pos_ref1_nm) * unit.nanometer
    )
    restraint_config.pos_ref2_nm = (
        floatstr2list(restraint_config.pos_ref2_nm) * unit.nanometer
    )
    ref_x1, ref_y1, ref_z1 = restraint_config.pos_ref1_nm
    ref_x2, ref_y2, ref_z2 = restraint_config.pos_ref2_nm
    k_r = restraint_config.restr_k * unit.kilojoules_per_mole
    vec_r = CustomCentroidBondForce(
        2,
        "(k{0}/2)*((x1-x2-ref_x1{0}+ref_x2{0})^2+(y1-y2-ref_y1{0}+ref_y2{0})^2+(z1-z2-ref_z1{0}+ref_z2{0})^2)".format(
            _name
        ),
    )
    vec_r.addGroup(restraint_config.vec_grp1)
    vec_r.addGroup(restraint_config.vec_grp2)
    vec_r.addBond([0, 1])
    vec_r.addGlobalParameter("k{}".format(_name), k_r)
    vec_r.addGlobalParameter("ref_x1{}".format(_name), ref_x1)
    vec_r.addGlobalParameter("ref_x2{}".format(_name), ref_x2)
    vec_r.addGlobalParameter("ref_y1{}".format(_name), ref_y1)
    vec_r.addGlobalParameter("ref_y2{}".format(_name), ref_y2)
    vec_r.addGlobalParameter("ref_z1{}".format(_name), ref_z1)
    vec_r.addGlobalParameter("ref_z2{}".format(_name), ref_z2)
    vec_r.setUsesPeriodicBoundaryConditions(True)
    return vec_r


def generate_upper_wall_restraint(restraint_config):
    # Upper wall.
    _name = restraint_config.name
    k1 = restraint_config.restr_k * unit.kilojoules_per_mole
    upper_wall = restraint_config["upper_wall_nm"] * unit.nanometer
    upper_wall_rest = CustomCentroidBondForce(
        3,
        "(k{0}/2)*max((distance(g1,g2)*(-cos(angle(g1,g2,g3)))) - upper_wall{0}, 0)^2".format(
            _name
        ),
    )
    upper_wall_rest.addGroup(restraint_config.restr_grp)
    upper_wall_rest.addGroup(restraint_config.gate_grp)
    upper_wall_rest.addGroup(restraint_config.pocket_grp)
    upper_wall_rest.addBond([0, 1, 2])
    upper_wall_rest.addGlobalParameter("k{}".format(_name), k1)
    upper_wall_rest.addGlobalParameter("upper_wall{}".format(_name), upper_wall)
    upper_wall_rest.setUsesPeriodicBoundaryConditions(True)
    return upper_wall_rest


def generate_side_wall_restraint(restraint_config):
    _name = restraint_config.name
    k2 = restraint_config.restr_k * unit.kilojoules_per_mole
    wall_width = restraint_config.width * unit.nanometer
    wall_buffer = restraint_config.buffer * unit.nanometer
    steepness = restraint_config.steepness * unit.nanometer
    s_center = restraint_config.s_center * unit.nanometer

    dist_restraint = CustomCentroidBondForce(
        3,
        "(k{0}/2)*max(distance(g1,g2)*sin(angle(g1,g2,g3)) - (a{0}/(1+exp(b{0}*(distance(g1,g2)*(-cos(angle(g1,g2,g3)))-c{0})))+d{0}), 0)^2".format(
            _name
        ),
    )
    dist_restraint.addGroup(restraint_config.restr_grp)
    dist_restraint.addGroup(restraint_config.gate_grp)
    dist_restraint.addGroup(restraint_config.pocket_grp)
    dist_restraint.addBond([0, 1, 2])
    dist_restraint.addGlobalParameter("k{}".format(_name), k2)
    dist_restraint.addGlobalParameter("a{}".format(_name), wall_width)
    dist_restraint.addGlobalParameter("b{}".format(_name), steepness)
    dist_restraint.addGlobalParameter("c{}".format(_name), s_center)
    dist_restraint.addGlobalParameter("d{}".format(_name), wall_buffer)
    dist_restraint.setUsesPeriodicBoundaryConditions(True)
    return dist_restraint


def generate_lower_wall_restraint(restraint_config):
    _name = restraint_config.name
    k1 = restraint_config.restr_k * unit.kilojoules_per_mole
    lower_wall = restraint_config["lower_wall_nm"] * unit.nanometer
    lower_wall_rest = CustomCentroidBondForce(
        3,
        "(k{0}/2)*max(distance(g1,g2)*cos(angle(g1,g2,g3)) - lower_wall{0}, 0)^2".format(
            _name
        ),
    )
    lower_wall_rest.addGroup(restraint_config.restr_grp)
    lower_wall_rest.addGroup(restraint_config.gate_grp)
    lower_wall_rest.addGroup(restraint_config.pocket_grp)
    lower_wall_rest.addBond([0, 1, 2])
    lower_wall_rest.addGlobalParameter("k{}".format(_name), k1)
    lower_wall_rest.addGlobalParameter("lower_wall{}".format(_name), lower_wall)
    lower_wall_rest.setUsesPeriodicBoundaryConditions(True)
    return lower_wall_rest


def generate_restraint_funnel(restraint_config):
    # restain in a sphere wall.
    # input: config
    # output: list of restraint [lower_wall,side_wall,upper_wall]
    restraint_config.restr_grp = idstr2list(restraint_config.restr_grp)
    restraint_config.gate_grp = idstr2list(restraint_config.gate_grp)
    restraint_config.pocket_grp = idstr2list(restraint_config.pocket_grp)

    lower_wall = generate_lower_wall_restraint(restraint_config)
    side_wall = generate_side_wall_restraint(restraint_config)
    upper_wall = generate_upper_wall_restraint(restraint_config)
    return [lower_wall, side_wall, upper_wall]


def generate_dist_min(restraint_config):
    # restain in a sphere wall.
    _name = restraint_config.name
    dis1_r = CustomCentroidBondForce(
        2, "(k{0}/2)*(max(dis1{0} - distance(g1,g2), 0)^order{0})".format(_name)
    )
    dis1_r.addGroup(restraint_config.grp1)
    dis1_r.addGroup(restraint_config.grp2)
    dis1_r.addBond([0, 1])
    dis1_r.addGlobalParameter("k{}".format(_name), restraint_config.restr_k)
    dis1_r.addGlobalParameter("dis1{}".format(_name), restraint_config.min_nm)
    dis1_r.addGlobalParameter("order{}".format(_name), restraint_config.order)
    dis1_r.setUsesPeriodicBoundaryConditions(True)
    return dis1_r


def generate_dist_max(restraint_config):
    # restain in a sphere wall.
    _name = restraint_config.name
    dis2_r = CustomCentroidBondForce(
        2, "(k{0}/2)*(max(distance(g1,g2) - dis2{0}, 0)^order{0})".format(_name)
    )
    dis2_r.addGroup(restraint_config.grp1)
    dis2_r.addGroup(restraint_config.grp2)
    dis2_r.addBond([0, 1])
    dis2_r.addGlobalParameter("k{}".format(_name), restraint_config.restr_k)
    dis2_r.addGlobalParameter("dis2{}".format(_name), restraint_config.max_nm)
    dis2_r.addGlobalParameter("order{}".format(_name), restraint_config.order)
    dis2_r.setUsesPeriodicBoundaryConditions(True)
    return dis2_r


def generate_restraint_distance(restraint_config):
    # restain in a sphere wall.
    # input: config
    # output: list of restraint [lower_wall,side_wall,upper_wall]
    restraint_config.grp1 = idstr2list(restraint_config.grp1)
    restraint_config.grp2 = idstr2list(restraint_config.grp2)
    restraint_config.restr_k = restraint_config.restr_k * unit.kilojoules_per_mole

    return_ls = []
    if restraint_config.get("min_nm"):
        restraint_config.min_nm = restraint_config.min_nm * unit.nanometer
        return_ls.append(generate_dist_min(restraint_config))
    if restraint_config.get("max_nm"):
        restraint_config.max_nm = restraint_config.max_nm * unit.nanometer
        return_ls.append(generate_dist_max(restraint_config))
    return return_ls


def generate_angle_min(restraint_config):
    # restain in a sphere wall.
    _name = restraint_config.name
    ang1_r = CustomCentroidBondForce(
        3, "(k{0}/2)*(max(ang1{0} - angle(g1, g2, g3), 0)^order{0})".format(_name)
    )
    ang1_r.addGroup(restraint_config.grp1)
    ang1_r.addGroup(restraint_config.grp2)
    ang1_r.addGroup(restraint_config.grp3)
    ang1_r.addBond([0, 1, 2])
    ang1_r.addGlobalParameter("k{}".format(_name), restraint_config.restr_k)
    ang1_r.addGlobalParameter("ang1{}".format(_name), restraint_config.min_degree)
    ang1_r.addGlobalParameter("order{}".format(_name), restraint_config.get("order", 2))
    ang1_r.setUsesPeriodicBoundaryConditions(True)
    return ang1_r


def generate_angle_max(restraint_config):
    # restain in a sphere wall.
    _name = restraint_config.name
    ang2_r = CustomCentroidBondForce(
        3, "(k{0}/2)*(max(angle(g1, g2, g3) - ang2{0}, 0)^order{0})".format(_name)
    )
    ang2_r.addGroup(restraint_config.grp1)
    ang2_r.addGroup(restraint_config.grp2)
    ang2_r.addGroup(restraint_config.grp3)
    ang2_r.addBond([0, 1, 2])
    ang2_r.addGlobalParameter("k{}".format(_name), restraint_config.restr_k)
    ang2_r.addGlobalParameter("ang2{}".format(_name), restraint_config.max_degree)
    ang2_r.addGlobalParameter("order{}".format(_name), restraint_config.get("order", 2))
    ang2_r.setUsesPeriodicBoundaryConditions(True)
    return ang2_r


def generate_restraint_angle(restraint_config):
    # restain in a sphere wall.
    # input: config
    # output: list of restraint [lower_wall,side_wall,upper_wall]
    restraint_config.grp1 = idstr2list(restraint_config.grp1)
    restraint_config.grp2 = idstr2list(restraint_config.grp2)
    restraint_config.grp3 = idstr2list(restraint_config.grp3)
    restraint_config.restr_k = restraint_config.restr_k * unit.kilojoules_per_mole

    return_ls = []
    if restraint_config.get("min_degree"):
        restraint_config.min_degree = restraint_config.min_degree * unit.degree
        return_ls.append(generate_angle_min(restraint_config))
    if restraint_config.get("max_degree"):
        restraint_config.max_degree = restraint_config.max_degree * unit.degree
        return_ls.append(generate_angle_max(restraint_config))
    return return_ls


def generate_restraint_dihedral(restraint_config):
    _name = restraint_config.name
    restraint_config.min_degree = fix_degree_range(restraint_config.min_degree)
    restraint_config.max_degree = fix_degree_range(restraint_config.max_degree)
    restraint_config.restr_k = restraint_config.restr_k * unit.kilojoules_per_mole

    if restraint_config.min_degree <= restraint_config.max_degree:
        dih_r = CustomCentroidBondForce(
            4,
            "k{0} * min(dihedral(g1, g2, g3,g4) - min_dih{0},0)^order{0} + \
                k{0} * max(dihedral(g1, g2, g3,g4) - max_dih{0},0)^order{0}".format(
                _name
            ),
        )

    else:
        dih_r = CustomCentroidBondForce(
            4,
            "min(k{0} * min(dihedral(g1, g2, g3,g4) - min_dih{0},0)^order{0}, \
                    k{0} * max(dihedral(g1, g2, g3,g4) - max_dih{0},0)^2order{0} )".format(
                _name
            ),
        )
    dih_r.addGroup(idstr2list(restraint_config.grp1))
    dih_r.addGroup(idstr2list(restraint_config.grp2))
    dih_r.addGroup(idstr2list(restraint_config.grp3))
    dih_r.addGroup(idstr2list(restraint_config.grp4))
    dih_r.addBond([0, 1, 2, 3])
    dih_r.addGlobalParameter("k{}".format(_name), restraint_config.restr_k)
    dih_r.addGlobalParameter("min_dih{}".format(_name), restraint_config.min_degree)
    dih_r.addGlobalParameter("max_dih{}".format(_name), restraint_config.max_degree)
    dih_r.addGlobalParameter("order{}".format(_name), restraint_config.get("order", 2))
    dih_r.setUsesPeriodicBoundaryConditions(True)
    return dih_r


def generate_min_x_restraint(restraint_config):
    # restain in a sphere wall.
    _name = restraint_config.name
    min_x = restraint_config.min_x_nm * unit.nanometer
    dis1_r = CustomCentroidBondForce(
        1, "(k{0}/2)*(min(x1-min_x{0}, 0)^order{0})".format(_name)
    )
    dis1_r.addBond([0])
    dis1_r.addGroup(restraint_config.restr_grp)
    dis1_r.addGlobalParameter("k{}".format(_name), restraint_config.restr_k)
    dis1_r.addGlobalParameter("min_x{}".format(_name), min_x)
    dis1_r.addGlobalParameter("order{}".format(_name), restraint_config.get("order", 2))
    dis1_r.setUsesPeriodicBoundaryConditions(False)
    return dis1_r


def generate_max_x_restraint(restraint_config):
    # restain in a sphere wall.
    _name = restraint_config.name
    max_x = restraint_config.max_x_nm * unit.nanometer
    dis1_r = CustomCentroidBondForce(
        1, "(k{0}/2)*(max(x1-max_x{0}, 0)^order{0})".format(_name)
    )
    dis1_r.addBond([0])
    dis1_r.addGroup(restraint_config.restr_grp)
    dis1_r.addGlobalParameter("k{}".format(_name), restraint_config.restr_k)
    dis1_r.addGlobalParameter("max_x{}".format(_name), max_x)
    dis1_r.addGlobalParameter("order{}".format(_name), restraint_config.get("order", 2))
    dis1_r.setUsesPeriodicBoundaryConditions(False)
    return dis1_r


def generate_min_y_restraint(restraint_config):
    # restain in a sphere wall.
    _name = restraint_config.name
    min_y = restraint_config.min_y_nm * unit.nanometer
    dis1_r = CustomCentroidBondForce(
        1, "(k{0}/2)*(min(y1-min_y{0}, 0)^2)".format(_name)
    )
    dis1_r.addBond([0])
    dis1_r.addGroup(restraint_config.restr_grp)
    dis1_r.addGlobalParameter("k{}".format(_name), restraint_config.restr_k)
    dis1_r.addGlobalParameter("min_y{}".format(_name), min_y)
    dis1_r.setUsesPeriodicBoundaryConditions(False)
    return dis1_r


def generate_max_y_restraint(restraint_config):
    # restain in a sphere wall.
    _name = restraint_config.name
    max_y = restraint_config.max_y_nm * unit.nanometer
    dis1_r = CustomCentroidBondForce(
        1, "(k{0}/2)*(max(y1-max_y{0}, 0)^2)".format(_name)
    )
    dis1_r.addBond([0])
    dis1_r.addGroup(restraint_config.restr_grp)
    dis1_r.addGlobalParameter("k{}".format(_name), restraint_config.restr_k)
    dis1_r.addGlobalParameter("max_y{}".format(_name), max_y)
    dis1_r.setUsesPeriodicBoundaryConditions(False)
    return dis1_r


def generate_min_z_restraint(restraint_config):
    # restain in a sphere wall.
    _name = restraint_config.name
    min_z = restraint_config.min_z_nm * unit.nanometer
    dis1_r = CustomCentroidBondForce(
        1, "(k{0}/2)*(min(z1-min_z{0}, 0)^2)".format(_name)
    )
    dis1_r.addBond([0])
    dis1_r.addGroup(restraint_config.restr_grp)
    dis1_r.addGlobalParameter("k{}".format(_name), restraint_config.restr_k)
    dis1_r.addGlobalParameter("min_z{}".format(_name), min_z)
    dis1_r.setUsesPeriodicBoundaryConditions(False)
    return dis1_r


def generate_max_z_restraint(restraint_config):
    # restain in a sphere wall.
    _name = restraint_config.name
    max_z = restraint_config.max_z_nm * unit.nanometer
    dis1_r = CustomCentroidBondForce(
        1, "(k{0}/2)*(max(z1-max_z{0}, 0)^2)".format(_name)
    )
    dis1_r.addBond([0])
    dis1_r.addGroup(restraint_config.restr_grp)
    dis1_r.addGlobalParameter("k{}".format(_name), restraint_config.restr_k)
    dis1_r.addGlobalParameter("max_z{}".format(_name), max_z)
    dis1_r.setUsesPeriodicBoundaryConditions(False)
    return dis1_r


def generate_xyz_box(restraint_config):
    restraint_config.restr_grp = idstr2list(restraint_config.restr_grp)
    restraint_config.restr_k = restraint_config.restr_k * unit.kilojoules_per_mole

    return_ls = []
    if restraint_config.get("min_x_nm"):
        return_ls.append(generate_min_x_restraint(restraint_config))
    if restraint_config.get("max_x_nm"):
        return_ls.append(generate_max_x_restraint(restraint_config))
    if restraint_config.get("min_y_nm"):
        return_ls.append(generate_min_y_restraint(restraint_config))
    if restraint_config.get("max_y_nm"):
        return_ls.append(generate_max_y_restraint(restraint_config))
    if restraint_config.get("min_z_nm"):
        return_ls.append(generate_min_z_restraint(restraint_config))
    if restraint_config.get("max_z_nm"):
        return_ls.append(generate_max_z_restraint(restraint_config))
    return return_ls


def generate_ref_position_min_restraint(restraint_config):
    # restain in a sphere wall.
    _name = restraint_config.name
    ref_pos = restraint_config.ref_position_nm
    dis1_r = CustomExternalForce(
        1,
        "0.5*k{0}*min(((x1-x0{0})^2+(y1-y0{0})^2+(z1-z0{0})^2)^0.5-min_dis{0},0)^order{0}".format(
            _name
        ),
    )
    dis1_r.addBond([0])
    dis1_r.addGroup(restraint_config.restr_grp)
    dis1_r.addGlobalParameter("k{}".format(_name), restraint_config.restr_k)
    dis1_r.addGlobalParameter("x0{}".format(_name), ref_pos[0])
    dis1_r.addGlobalParameter("y0{}".format(_name), ref_pos[1])
    dis1_r.addGlobalParameter("z0{}".format(_name), ref_pos[2])
    dis1_r.addGlobalParameter("min_dis{}".format(_name), restraint_config.min_nm)
    dis1_r.addGlobalParameter("order{}".format(_name), restraint_config.get("order", 2))
    # dis1_r.setUsesPeriodicBoundaryConditions(False)
    return dis1_r


def generate_ref_position_max_restraint(restraint_config):
    # restain in a sphere wall.
    _name = restraint_config.name
    ref_pos = restraint_config.ref_position_nm
    dis1_r = CustomCentroidBondForce(
        1,
        "0.5*k{0}*max(((x1-x0{0})^2+(y1-y0{0})^2+(z1-z0{0})^2)^0.5-max_dis{0},0)^order{0}".format(
            _name
        ),
    )
    dis1_r.addBond([0])
    dis1_r.addGroup(restraint_config.restr_grp)
    dis1_r.addGlobalParameter("k{}".format(_name), restraint_config.restr_k)
    dis1_r.addGlobalParameter("x0{}".format(_name), ref_pos[0])
    dis1_r.addGlobalParameter("y0{}".format(_name), ref_pos[1])
    dis1_r.addGlobalParameter("z0{}".format(_name), ref_pos[2])
    dis1_r.addGlobalParameter("max_dis{}".format(_name), restraint_config.max_nm)
    dis1_r.addGlobalParameter("order{}".format(_name), restraint_config.get("order", 2))
    # dis1_r.setUsesPeriodicBoundaryConditions(True)
    return dis1_r


# need system when add virtual particle
def generate_dist_ref_position(restraint_config):
    # restain in a sphere wall.
    # input: config
    # output: list of restraint [lower_wall,side_wall,upper_wall]
    # simulation.reporters[-1].get_restraint_dist_ref_position(simulation=simulation, restraint_config=simulation.reporters[-1].restraint_config['restr_A'])
    restraint_config.restr_grp = idstr2list(restraint_config.restr_grp)
    restraint_config.ref_position_nm = (
        floatstr2list(restraint_config.ref_position_nm) * unit.nanometer
    )
    if restraint_config.get("restr_k_per_atom"):
        restraint_config.restr_k = (
            restraint_config.restr_k_per_atom
            * len(restraint_config.restr_grp)
            * unit.kilojoules_per_mole
        )
    else:
        restraint_config.restr_k = restraint_config.restr_k * unit.kilojoules_per_mole

    return_ls = []
    if restraint_config.get("min_nm"):
        restraint_config.min_nm = restraint_config.min_nm * unit.nanometer
        return_ls.append(generate_ref_position_min_restraint(restraint_config))
    if restraint_config.get("max_nm"):
        restraint_config.max_nm = restraint_config.max_nm * unit.nanometer
        return_ls.append(generate_ref_position_max_restraint(restraint_config))

    return return_ls


def fix_degree_range(in_degree):
    x = in_degree
    x = np.mod(x, 360)
    if x > 180:
        x = x - 360
    return x * unit.degree


def force_test(restraint_config):
    _name = restraint_config.name
    restraint_config.min_dih_degree = fix_degree_range(restraint_config.min_dih_degree)
    restraint_config.max_dih_degree = fix_degree_range(restraint_config.max_dih_degree)
    restraint_config.restr_k = restraint_config.restr_k * unit.kilojoules_per_mole

    if restraint_config.min_dih_degree <= restraint_config.max_dih_degree:
        dih_r = CustomCentroidBondForce(
            4,
            "k{0} * min(dihedral(g1, g2, g3,g4) - min_dih{0},0)^2 + \
                k{0} * max(dihedral(g1, g2, g3,g4) - max_dih{0},0)^2".format(
                _name
            ),
        )

    else:
        dih_r = CustomCentroidBondForce(
            4,
            "min(k{0} * min(dihedral(g1, g2, g3,g4) - min_dih{0},0)^2, \
                    k{0} * max(dihedral(g1, g2, g3,g4) - max_dih{0},0)^2 )".format(
                _name
            ),
        )
    dih_r.addGroup(idstr2list(restraint_config.grp1))
    dih_r.addGroup(idstr2list(restraint_config.grp2))
    dih_r.addGroup(idstr2list(restraint_config.grp3))
    dih_r.addGroup(idstr2list(restraint_config.grp4))
    dih_r.addBond([0, 1, 2, 3])
    dih_r.addGlobalParameter("k{}".format(_name), restraint_config.restr_k)
    dih_r.addGlobalParameter("min_dih{}".format(_name), restraint_config.min_dih_degree)
    dih_r.addGlobalParameter("max_dih{}".format(_name), restraint_config.max_dih_degree)
    dih_r.setUsesPeriodicBoundaryConditions(True)
    return dih_r
