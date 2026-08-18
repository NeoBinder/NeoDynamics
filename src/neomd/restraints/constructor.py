from openmm import unit
from openmm.openmm import CustomCentroidBondForce

def generate_restraint(restraint_config):
    if restraint_config["type"] == "funnel":
        restraint = generate_restraint_funnel(restraint_config)
    elif restraint_config["type"] == "distance":
        restraint = generate_restraint_distance(restraint_config)
    elif restraint_config["type"] == "distances":
        restraint = generate_restraint_distances(restraint_config)
    elif restraint_config["type"] == "angle":
        restraint = generate_restraint_angle(restraint_config)
    elif restraint_config["type"] == "dihedral":
        restraint = generate_restraint_dihedral(restraint_config)
    elif restraint_config["type"] == "dist_ref_position":
        restraint = generate_dist_ref_position(restraint_config)
    elif restraint_config['type'] == 'rmsd':
        restraint = generate_restraint_rmsd(restraint_config)
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
    if type(bond_info["params"])==dict:
        _bond = CustomCentroidBondForce(len(grps), func)
        grp_ids=[]
        for i,grp in enumerate(grps):
            _bond.addGroup(grp)
            grp_ids.append(i)
        _params=[]
        for k, v in bond_info["params"].items():
            _bond.addPerBondParameter(k)
            _params.append(v)
        _bond.addBond(grp_ids, _params)
    elif type(bond_info["params"])==list:
        _bond = CustomCentroidBondForce(len(grps[0]), func)
        grps_dic={}
        i=0
        for _grp in grps:
            for grp in _grp:
                if grp not in grps_dic.values():
                    grps_dic[i]=grp
                    i+=1
                    _bond.addGroup(grp)
        param=bond_info["params"][0]
        for k, v in param.items():
            _bond.addPerBondParameter(k)
        for i,param in enumerate(bond_info["params"]):
            _params=[]
            for param_i in range(_bond.getNumPerBondParameters()):
                param_name=_bond.getPerBondParameterName(param_i)
                _params.append(param[param_name])
            _grps=bond_info["grps"][i]
            __grps=[next((k for k, v in grps_dic.items() if v == grp), None) for grp in _grps]
            _bond.addBond(__grps, _params)
    else:
        raise ValueError("params should be either dict or list of dict")
    _bond.setUsesPeriodicBoundaryConditions(bond_info["is_periodic"])
    return _bond



def generate_vec_restraint(restraint_config):
    # restraint 2 groups of atoms to a defined vector
    # restraint both direction and length of the vector
    restraint_config.pos_ref1_nm = (
        restraint_config.pos_ref1_nm * unit.nanometer
    )
    restraint_config.pos_ref2_nm = (
        restraint_config.pos_ref2_nm * unit.nanometer
    )
    ref_x1, ref_y1, ref_z1 = restraint_config.pos_ref1_nm
    ref_x2, ref_y2, ref_z2 = restraint_config.pos_ref2_nm
    info = {
        "grps": [restraint_config.vec_grp1, restraint_config.vec_grp2],
        "func": "(k/2)*((x1-x2-ref_x1+ref_x2)^2+(y1-y2-ref_y1+ref_y2)^2+(z1-z2-ref_z1+ref_z2)^2)",
        "params": {
            "k": restraint_config.restr_k * unit.kilojoules_per_mole,
            "ref_x1": ref_x1,
            "ref_x2": ref_x2,
            "ref_y1": ref_y1,
            "ref_y2": ref_y2,
            "ref_z1": ref_z1,
            "ref_z2": ref_z2,
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
            "func": "(k/2)*max((distance(g1,g2)*(-cos(angle(g1,g2,g3)))) - upper_wall, 0)^2",
            "params": {
                "k": restraint_config.restr_k,
                "upper_wall": restraint_config["upper_wall_nm"]
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
            "func": "(k/2)*max(distance(g1,g2)*sin(angle(g1,g2,g3)) - (a/(1+exp(b*(distance(g1,g2)*(-cos(angle(g1,g2,g3)))-c)))+d), 0)^2",
            "params": {
                "k": restraint_config.restr_k,
                "a": restraint_config.width * unit.nanometer,  # wall_width
                "b": restraint_config.steepness * unit.nanometer,  # steepness
                "c": restraint_config.s_center * unit.nanometer,  # s_center
                "d": restraint_config.buffer * unit.nanometer,  # wall_buffer
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
            "func": "(k/2)*max(distance(g1,g2)*cos(angle(g1,g2,g3)) - lower_wall, 0)^2",
            "params": {
                "k": restraint_config.restr_k,
                "lower_wall": restraint_config.lower_wall_nm * unit.nanometer,
            },
            "is_periodic": restraint_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info)

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
            "func": "(k/2)*(max(dis1 - distance(g1,g2), 0)^order)",
            "params": {
                "k": restraint_config.restr_k,
                "dis1": restraint_config.min_nm * unit.nanometer,
                "order": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info_min)

    def generate_dist_max(restraint_config):
        info_max = {
            "grps": [restraint_config.grp1, restraint_config.grp2],
            "func": "(k/2)*(max(distance(g1,g2) - dis2, 0)^order)",
            "params": {
                "k": restraint_config.restr_k,
                "dis2": restraint_config.max_nm * unit.nanometer,
                "order": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info_max)

    restraint_config.restr_k = restraint_config.restr_k * unit.kilojoules_per_mole

    return_ls = []
    if restraint_config.get("min_nm") != None:
        return_ls.append(generate_dist_min(restraint_config))
    if restraint_config.get("max_nm") != None:
        return_ls.append(generate_dist_max(restraint_config))
    return return_ls

def generate_restraint_distances(restraint_config):
    # restraint the distance between two groups of atoms
    # if min_nm is defined, add a restraint when distance is smaller than min_nm
    # if max_nm is defined, add a restraint when distance is larger than max_nm
    def generate_dist_min(restraint_config):
        info_min = {
            "func": "(k/2)*(max(dis1 - distance(g1,g2), 0)^order)",
            "grps": [],
            "params": [],
            "is_periodic": restraint_config.get("is_periodic", True),
        }
        for param in restraint_config.params:
            if param.get("min_nm") != None:
                info_min["grps"].append([param.grp1, param.grp2])
                info_min["params"].append({
                        "k": param.restr_k * unit.kilojoules_per_mole,
                        "dis1": param.min_nm * unit.nanometer,
                        "order": param.get("order", 2),
                    })
        return generate_CustomCentroidBondForce(info_min) if len(info_min["params"]) > 0 else None

    def generate_dist_max(restraint_config):
        info_max = {
            "func": "(k/2)*(max(distance(g1,g2) - dis2, 0)^order)",
            "grps": [],
            "params": [],
            "is_periodic": restraint_config.get("is_periodic", True),
        }
        for param in restraint_config.params:
            if param.get("max_nm") != None:
                info_max["grps"].append([param.grp1, param.grp2])
                info_max["params"].append({
                        "k": param.restr_k * unit.kilojoules_per_mole,
                        "dis2": param.max_nm * unit.nanometer,
                        "order": param.get("order", 2),
                    })
        return generate_CustomCentroidBondForce(info_max) if len(info_max["params"]) > 0 else None
    return_ls = []
    dist_min=generate_dist_min(restraint_config)
    if dist_min is not None:
        return_ls.append(dist_min)
    dist_max=generate_dist_max(restraint_config)
    if dist_max is not None:
        return_ls.append(dist_max)
    return return_ls

def generate_restraint_angle(restraint_config):
    # restraint the angle between three groups of atoms
    # if min_degree is defined, add a restraint when angle is smaller than min_degree
    # if max_degree is defined, add a restraint when angle is larger than max_degree
    def generate_angle_min(restraint_config):
        info = {
            "grps": [
                restraint_config.grp1,
                restraint_config.grp2,
                restraint_config.grp3,
            ],
            "func": "(k/2)*(max(ang1 - angle(g1, g2, g3), 0)^order)",
            "params": {
                "k": restraint_config.restr_k,
                "ang1": restraint_config.min_degree * unit.degree,
                "order": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_angle_max(restraint_config):
        info = {
            "grps": [
                restraint_config.grp1,
                restraint_config.grp2,
                restraint_config.grp3,
            ],
            "func": "(k/2)*(max(angle(g1, g2, g3) - ang2, 0)^order)",
            "params": {
                "k": restraint_config.restr_k,
                "ang2": restraint_config.max_degree * unit.degree,
                "order": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info)

    restraint_config.restr_k = restraint_config.restr_k * unit.kilojoules_per_mole

    return_ls = []
    if restraint_config.get("min_degree") != None:
        return_ls.append(generate_angle_min(restraint_config))
    if restraint_config.get("max_degree") != None:
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

    restraint_config.max_degree = fix_max_angle(
        restraint_config.min_degree, restraint_config.max_degree
    )
    arctan_x = "atan(tan((dihedral(g1,g2,g3,g4)-(min_dih+max_dih)/2)/2))"
    arctan_half_diff = "atan(tan((max_dih - min_dih)/4))"
    energy_min = f"abs(min({arctan_x} - (-({arctan_half_diff})), 0))"
    energy_max = f"abs(max({arctan_x} - {arctan_half_diff}, 0))"
    info = {
        "grps": [
            restraint_config.grp1,
            restraint_config.grp2,
            restraint_config.grp3,
            restraint_config.grp4,
        ],
        "func": f"k*({energy_min}+{energy_max})^order",
        "params": {
            "k": restraint_config.restr_k * unit.kilojoules_per_mole,
            "min_dih": restraint_config.min_degree * unit.degree,
            "max_dih": restraint_config.max_degree * unit.degree,
            "order": restraint_config.get("order", 2),
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
            "func": "(k/2)*(min(x1-min_x, 0)^order)",
            "params": {
                "k": restraint_config.restr_k,
                "min_x": restraint_config.min_x_nm * unit.nanometer,
                "order": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get(
                "is_periodic", False
            ),  # 坐标约束通常不需要周期边界
        }
        return generate_CustomCentroidBondForce(info)

    def generate_max_x_restraint(restraint_config):
        info = {
            "grps": [restraint_config.restr_grp],
            "func": "(k/2)*(max(x1-max_x, 0)^order)",
            "params": {
                "k": restraint_config.restr_k,
                "max_x": restraint_config.max_x_nm * unit.nanometer,
                "order": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_min_y_restraint(restraint_config):
        info = {
            "grps": [restraint_config.restr_grp],
            "func": "(k/2)*(min(y1-min_y, 0)^2)",
            "params": {
                "k": restraint_config.restr_k,
                "min_y": restraint_config.min_y_nm * unit.nanometer,
                "order": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_max_y_restraint(restraint_config):
        info = {
            "grps": [restraint_config.restr_grp],
            "func": "(k/2)*(max(y1-max_y, 0)^2)",
            "params": {
                "k": restraint_config.restr_k,
                "max_y": restraint_config.max_y_nm * unit.nanometer,
                "order": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_min_z_restraint(restraint_config):
        info = {
            "grps": [restraint_config.restr_grp],
            "func": "(k/2)*(min(z1-min_z, 0)^2)",
            "params": {
                "k": restraint_config.restr_k,
                "min_z": restraint_config.min_z_nm * unit.nanometer,
                "order": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_max_z_restraint(restraint_config):
        info = {
            "grps": [restraint_config.restr_grp],
            "func": "(k/2)*(max(z1-max_z, 0)^2)",
            "params": {
                "k": restraint_config.restr_k,
                "max_z": restraint_config.max_z_nm * unit.nanometer,
                "order": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

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
            "func": "0.5*k*min(((x1-x0)^2+(y1-y0)^2+(z1-z0)^2)^0.5-min_dis,0)^order",
            "params": {
                "k": restraint_config.restr_k,
                "x0": ref_pos[0],
                "y0": ref_pos[1],
                "z0": ref_pos[2],
                "min_dis": restraint_config.min_nm * unit.nanometer,
                "order": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_ref_position_max_restraint(restraint_config):
        info = {
            "grps": [restraint_config.restr_grp],
            "func": "0.5*k*max(((x1-x0)^2+(y1-y0)^2+(z1-z0)^2)^0.5-max_dis,0)^order",
            "params": {
                "k": restraint_config.restr_k,
                "x0": ref_pos[0],
                "y0": ref_pos[1],
                "z0": ref_pos[2],
                "max_dis": restraint_config.max_nm * unit.nanometer,
                "order": restraint_config.get("order", 2),
            },
            "is_periodic": restraint_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

    restraint_config.ref_position_nm = (
        restraint_config.ref_position_nm * unit.nanometer
    )
    if restraint_config.get("restr_k_per_atom"):
        restraint_config.restr_k = (
            restraint_config.restr_k_per_atom
            * len(restraint_config.restr_grp)
            * unit.kilojoules_per_mole
        )
    else:
        restraint_config.restr_k = restraint_config.restr_k * unit.kilojoules_per_mole

    ref_pos = restraint_config.ref_position_nm
    return_ls = []
    if restraint_config.get("min_nm") != None:
        return_ls.append(generate_ref_position_min_restraint(restraint_config))
    if restraint_config.get("max_nm") != None:
        return_ls.append(generate_ref_position_max_restraint(restraint_config))

    return return_ls

def generate_restraint_rmsd(restraint_config):
    import openmm
    from openmm.app import PDBxFile, PDBFile
    
    if restraint_config.ref_pos_file.endswith('.pdbx'):
        pos=PDBxFile(restraint_config.ref_pos_file).positions
    elif restraint_config.ref_pos_file.endswith('.pdb'):
        pos=PDBFile(restraint_config.ref_pos_file).positions
    else:
        raise ValueError(f'ref_pos_file should be pdb or pdbx, {restraint_config.ref_pos_file} is not either')

    rmsd_cv = openmm.RMSDForce(pos,
                               restraint_config.restr_grp)

    _name=restraint_config.name
    maxRMSD = restraint_config.maxRMSD_nm * unit.nanometer
    k_r = restraint_config.restr_k * unit.kilojoules_per_mole * len(restraint_config.restr_grp)
    k_name=rename_global_parameter('k'+_name)
    RMSD_name=rename_global_parameter('RMSD'+_name)
    maxRMSD_name=rename_global_parameter('maxRMSD'+_name)
    energy_expression = f"({k_name}/2)*max(0, {RMSD_name} - {maxRMSD_name})^2"
    restraint_force = openmm.CustomCVForce(energy_expression)
    restraint_force.addCollectiveVariable(RMSD_name, rmsd_cv)
    restraint_force.addGlobalParameter(maxRMSD_name, maxRMSD)
    restraint_force.addGlobalParameter(k_name, k_r)
    restraint_config['_force']=restraint_force
    return restraint_force

def rename_global_parameter(parameter_name):
    return f'restraint_{parameter_name}'