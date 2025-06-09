from openmm import unit
from openmm.openmm import CustomCentroidBondForce
from neomd.utils import floatstr2list, idstr2list


def generate_restraint(restraint_config, system=None):
    if restraint_config["type"] == "funnel":
        restraint = generate_restraint_funnel(restraint_config)
    elif restraint_config["type"] == "distance":
        restraint = generate_restraint_distance(restraint_config)
    elif restraint_config["type"] == "angle":
        restraint = generate_restraint_angle(restraint_config)
    elif restraint_config["type"] == "dihedral":
        restraint = generate_restraint_dihedral(restraint_config)
    # elif restraint_config["type"] == "ref_file":
    #     restraint = generate_restraint_ref_file(restraint_config, system=system)
    elif restraint_config["type"] == "dist_ref_position":
        restraint = generate_dist_ref_position(restraint_config)
    elif restraint_config["type"] == "xyz_box":
        restraint = generate_xyz_box(restraint_config)
    elif restraint_config["type"] == "vec_restraint":
        restraint = generate_vec_restraint(restraint_config)
    else:
        raise NotImplementedError(
            "restraint type:{} not defined".format(restraint_config["type"])
        )
    return restraint


def generate_CustomCentroidBondForce(bond_info):
    # base custom bond function
    # position calculation based on the center of mass of each atom group
    grps = bond_info["grps"]
    func = bond_info["func"]
    _bond = CustomCentroidBondForce(len(grps), func)
    for grp in bond_info["grps"]:
        _bond.addGroup(grp)
    _bond.addBond(list(range(len(grps))))
    for k, v in bond_info["params"].items():
        _bond.addGlobalParameter(k, v)
    _bond.setUsesPeriodicBoundaryConditions(bond_info["is_periodic"])
    return _bond


def generate_vec_restraint(restraint_config):
    # restraint 2 groups of atoms to a defined vector
    # restraint both direction and length of the vector
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
    info = {
        "grps": [restraint_config.vec_grp1, restraint_config.vec_grp2],
        "func": "(k{0}/2)*((x1-x2-ref_x1{0}+ref_x2{0})^2+(y1-y2-ref_y1{0}+ref_y2{0})^2+(z1-z2-ref_z1{0}+ref_z2{0})^2)".format(
            _name
        ),
        "params": {
            f"k{_name}": restraint_config.restr_k * unit.kilojoules_per_mole,
            f"ref_x1{_name}": ref_x1,
            f"ref_x2{_name}": ref_x2,
            f"ref_y1{_name}": ref_y1,
            f"ref_y2{_name}": ref_y2,
            f"ref_z1{_name}": ref_z1,
            f"ref_z2{_name}": ref_z2,
        },
        "is_periodic": restraint_config.get("is_periodic", True),
    }
    return generate_CustomCentroidBondForce(info)


def generate_restraint_funnel(restraint_config):
    # restrain in a funnel-shape wall.
    # input: config
    # output: list of restraint [lower_wall,side_wall,upper_wall]
    def generate_upper_wall_restraint(restraint_config):
        info = {
            "grps": [
                restraint_config.restr_grp,
                restraint_config.gate_grp,
                restraint_config.pocket_grp,
            ],
            "func": "(k{0}/2)*max((distance(g1,g2)*(-cos(angle(g1,g2,g3)))) - upper_wall{0}, 0)^2".format(
                _name
            ),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"upper_wall{_name}": restraint_config["upper_wall_nm"]
                * unit.nanometer,
            },
            "is_periodic": restraint_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_side_wall_restraint(restraint_config):
        info = {
            "grps": [
                restraint_config.restr_grp,
                restraint_config.gate_grp,
                restraint_config.pocket_grp,
            ],
            "func": "(k{0}/2)*max(distance(g1,g2)*sin(angle(g1,g2,g3)) - (a{0}/(1+exp(b{0}*(distance(g1,g2)*(-cos(angle(g1,g2,g3)))-c{0})))+d{0}), 0)^2".format(
                _name
            ),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"a{_name}": restraint_config.width * unit.nanometer,  # wall_width
                f"b{_name}": restraint_config.steepness * unit.nanometer,  # steepness
                f"c{_name}": restraint_config.s_center * unit.nanometer,  # s_center
                f"d{_name}": restraint_config.buffer * unit.nanometer,  # wall_buffer
            },
            "is_periodic": restraint_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_lower_wall_restraint(restraint_config):
        info = {
            "grps": [
                restraint_config.restr_grp,
                restraint_config.gate_grp,
                restraint_config.pocket_grp,
            ],
            "func": "(k{0}/2)*max(distance(g1,g2)*cos(angle(g1,g2,g3)) - lower_wall{0}, 0)^2".format(
                _name
            ),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"lower_wall{_name}": restraint_config.lower_wall_nm * unit.nanometer,
            },
            "is_periodic": restraint_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info)

    restraint_config.restr_grp = idstr2list(restraint_config.restr_grp)
    restraint_config.gate_grp = idstr2list(restraint_config.gate_grp)
    restraint_config.pocket_grp = idstr2list(restraint_config.pocket_grp)

    _name = restraint_config.name
    restraint_config.restr_k = restraint_config.restr_k * unit.kilojoules_per_mole
    lower_wall = generate_lower_wall_restraint(restraint_config)
    side_wall = generate_side_wall_restraint(restraint_config)
    upper_wall = generate_upper_wall_restraint(restraint_config)
    return [lower_wall, side_wall, upper_wall]


def generate_restraint_distance(restraint_config):
    # restraint the distance between two groups of atoms
    # if min_nm is defined, add a restraint when distance is smaller than min_nm
    # if max_nm is defined, add a restraint when distance is larger than max_nm
    def generate_dist_min(restraint_config):
        info_min = {
            "grps": [restraint_config.grp1, restraint_config.grp2],
            "func": "(k{0}/2)*(max(dis1{0} - distance(g1,g2), 0)^order{0})".format(
                _name
            ),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"dis1{_name}": restraint_config.min_nm * unit.nanometer,
                f"order{_name}": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info_min)

    def generate_dist_max(restraint_config):
        info_max = {
            "grps": [restraint_config.grp1, restraint_config.grp2],
            "func": "(k{0}/2)*(max(distance(g1,g2) - dis2{0}, 0)^order{0})".format(
                _name
            ),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"dis2{_name}": restraint_config.max_nm * unit.nanometer,
                f"order{_name}": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info_max)

    restraint_config.grp1 = idstr2list(restraint_config.grp1)
    restraint_config.grp2 = idstr2list(restraint_config.grp2)
    restraint_config.restr_k = restraint_config.restr_k * unit.kilojoules_per_mole

    return_ls = []
    _name = restraint_config.name
    if restraint_config.get("min_nm"):
        return_ls.append(generate_dist_min(restraint_config))
    if restraint_config.get("max_nm"):
        return_ls.append(generate_dist_max(restraint_config))
    return return_ls


def generate_restraint_angle(restraint_config):
    # restraint the angle between three groups of atoms
    # if min_degree is defined, add a restraint when angle is smaller than min_degree
    # if max_degree is defined, add a restraint when angle is larger than max_degree
    def generate_angle_min(restraint_config):
        _name = restraint_config.name
        info = {
            "grps": [
                restraint_config.grp1,
                restraint_config.grp2,
                restraint_config.grp3,
            ],
            "func": "(k{0}/2)*(max(ang1{0} - angle(g1, g2, g3), 0)^order{0})".format(
                _name
            ),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"ang1{_name}": restraint_config.min_degree * unit.degree,
                f"order{_name}": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_angle_max(restraint_config):
        _name = restraint_config.name
        info = {
            "grps": [
                restraint_config.grp1,
                restraint_config.grp2,
                restraint_config.grp3,
            ],
            "func": "(k{0}/2)*(max(angle(g1, g2, g3) - ang2{0}, 0)^order{0})".format(
                _name
            ),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"ang2{_name}": restraint_config.max_degree * unit.degree,
                f"order{_name}": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info)

    restraint_config.grp1 = idstr2list(restraint_config.grp1)
    restraint_config.grp2 = idstr2list(restraint_config.grp2)
    restraint_config.grp3 = idstr2list(restraint_config.grp3)
    restraint_config.restr_k = restraint_config.restr_k * unit.kilojoules_per_mole

    return_ls = []
    if restraint_config.get("min_degree"):
        return_ls.append(generate_angle_min(restraint_config))
    if restraint_config.get("max_degree"):
        return_ls.append(generate_angle_max(restraint_config))
    return return_ls


def generate_restraint_dihedral(restraint_config):
    # restraint the dihedral angle between four groups of atoms
    # min_degree and max_degree are always needed because the dihedral angle is periodic
    def fix_max_angle(min_angle, max_angle):
        # make max_angle always in the range of [min_angle, min_angle + 360)
        import math

        max_angle += 360 * math.ceil((min_angle - max_angle) / 360)
        return max_angle

    _name = restraint_config.name
    restraint_config.max_degree = fix_max_angle(
        restraint_config.min_degree, restraint_config.max_degree
    )

    arctan_x = f"atan(tan((dihedral(g1,g2,g3,g4)-(min_dih{_name}+max_dih{_name})/2)/2))"
    arctan_half_diff = f"atan(tan((max_dih{_name} - min_dih{_name})/4))"
    energy_min = f"abs(min({arctan_x} - (-({arctan_half_diff})), 0))"
    energy_max = f"abs(max({arctan_x} - {arctan_half_diff}, 0))"
    info = {
        "grps": [
            restraint_config.grp1,
            restraint_config.grp2,
            restraint_config.grp3,
            restraint_config.grp4,
        ],
        "func": f"k*({energy_min}+{energy_max})^order{_name}",
        "params": {
            f"k{_name}": restraint_config.restr_k * unit.kilojoules_per_mole,
            f"min_dih{_name}": restraint_config.min_degree * unit.degree,
            f"max_dih{_name}": restraint_config.max_degree * unit.degree,
            f"order{_name}": restraint_config.get("order", 2),
        },
        "is_periodic": restraint_config.get("is_periodic", True),
    }
    return generate_CustomCentroidBondForce(info)


def generate_xyz_box(restraint_config):
    # restraint the position of a group of atoms in a box defined by x, y, z range
    # not all of x, y, z are always needed, so we check the config for each direction
    def generate_min_x_restraint(restraint_config):
        info = {
            "grps": [restraint_config.restr_grp],
            "func": "(k{0}/2)*(min(x1-min_x{0}, 0)^order{0})".format(_name),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"min_x{_name}": restraint_config.min_x_nm * unit.nanometer,
                f"order{_name}": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get(
                "is_periodic", False
            ),  # 坐标约束通常不需要周期边界
        }
        return generate_CustomCentroidBondForce(info)

    def generate_max_x_restraint(restraint_config):
        info = {
            "grps": [restraint_config.restr_grp],
            "func": "(k{0}/2)*(max(x1-max_x{0}, 0)^order{0})".format(_name),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"max_x{_name}": restraint_config.max_x_nm * unit.nanometer,
                f"order{_name}": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_min_y_restraint(restraint_config):
        info = {
            "grps": [restraint_config.restr_grp],
            "func": "(k{0}/2)*(min(y1-min_y{0}, 0)^2)".format(_name),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"min_y{_name}": restraint_config.min_y_nm * unit.nanometer,
                f"order{_name}": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_max_y_restraint(restraint_config):
        info = {
            "grps": [restraint_config.restr_grp],
            "func": "(k{0}/2)*(max(y1-max_y{0}, 0)^2)".format(_name),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"max_y{_name}": restraint_config.max_y_nm * unit.nanometer,
                f"order{_name}": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_min_z_restraint(restraint_config):
        info = {
            "grps": [restraint_config.restr_grp],
            "func": "(k{0}/2)*(min(z1-min_z{0}, 0)^2)".format(_name),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"min_z{_name}": restraint_config.min_z_nm * unit.nanometer,
                f"order{_name}": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_max_z_restraint(restraint_config):
        info = {
            "grps": [restraint_config.restr_grp],
            "func": "(k{0}/2)*(max(z1-max_z{0}, 0)^2)".format(_name),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"max_z{_name}": restraint_config.max_z_nm * unit.nanometer,
                f"order{_name}": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

    restraint_config.restr_grp = idstr2list(restraint_config.restr_grp)
    _name = restraint_config.name
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


# need system when add virtual particle
def generate_dist_ref_position(restraint_config):
    # restraint the distance between a group of atoms and a reference position
    # if min_nm is defined, add a restraint when distance is smaller than min_nm
    # if max_nm is defined, add a restraint when distance is larger than max_nm
    def generate_ref_position_min_restraint(restraint_config):
        info = {
            "grps": [restraint_config.restr_grp],
            "func": "0.5*k{0}*min(((x1-x0{0})^2+(y1-y0{0})^2+(z1-z0{0})^2)^0.5-min_dis{0},0)^order{0}".format(
                _name
            ),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"x0{_name}": ref_pos[0],
                f"y0{_name}": ref_pos[1],
                f"z0{_name}": ref_pos[2],
                f"min_dis{_name}": restraint_config.min_nm * unit.nanometer,
                f"order{_name}": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_ref_position_max_restraint(restraint_config):
        info = {
            "grps": [restraint_config.restr_grp],
            "func": "0.5*k{0}*max(((x1-x0{0})^2+(y1-y0{0})^2+(z1-z0{0})^2)^0.5-max_dis{0},0)^order{0}".format(
                _name
            ),
            "params": {
                f"k{_name}": restraint_config.restr_k,
                f"x0{_name}": ref_pos[0],
                f"y0{_name}": ref_pos[1],
                f"z0{_name}": ref_pos[2],
                f"max_dis{_name}": restraint_config.max_nm * unit.nanometer,
                f"order{_name}": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

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

    _name = restraint_config.name
    ref_pos = restraint_config.ref_position_nm
    return_ls = []
    if restraint_config.get("min_nm"):
        return_ls.append(generate_ref_position_min_restraint(restraint_config))
    if restraint_config.get("max_nm"):
        return_ls.append(generate_ref_position_max_restraint(restraint_config))

    return return_ls
