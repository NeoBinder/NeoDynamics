from openmm import unit
from openmm.openmm import CustomCentroidBondForce


def generate_SMDforce(force_config):
    if force_config["type"] == "distance":
        restraint = generate_restraint_distance(force_config)
    elif force_config["type"] == "angle":
        restraint = generate_restraint_angle(force_config)
    elif force_config["type"] == "dihedral":
        restraint = generate_restraint_dihedral(force_config)
    elif force_config["type"] == "dist_ref_position":
        restraint = generate_dist_ref_position(force_config)
    elif force_config['type'] == 'rmsd':
        restraint = generate_restraint_rmsd(force_config)
    else:
        raise NotImplementedError(
            "restraint type:{} not defined".format(force_config["type"])
        )
    return restraint


def get_update_params(force_config,params_name):
    from collections import OrderedDict

    update_params = OrderedDict()
    for param in params_name:
        if force_config.get(param):
            assert isinstance(force_config[param], list)
            if len(force_config[param]) > 1:
                update_params[param] = force_config[param]
    return update_params
def generate_CustomCentroidBondForce(bond_info):
    # base custom bond function
    # position calculation based on the center of mass of each atom group
    grps = bond_info["grps"]
    func = bond_info["func"]
    _bond = CustomCentroidBondForce(len(grps), func)
    for grp in grps:
        _bond.addGroup(grp)
    _bond.addBond(list(range(len(grps))))
    for k, v in bond_info["params"].items():
        _bond.addGlobalParameter(k, v)
    _bond.setUsesPeriodicBoundaryConditions(bond_info["is_periodic"])
    return _bond

def generate_restraint_distance(force_config):
    # restraint the distance between two groups of atoms
    # if min_nm is defined, add a restraint when distance is smaller than min_nm
    # if max_nm is defined, add a restraint when distance is larger than max_nm
    def generate_dist_min(force_config):
        force_config.min_nm=[x * unit.nanometer for x in force_config.min_nm]
        info_min = {
            "grps": [force_config.grp1, force_config.grp2],
            "func": "(restr_k{0}/2)*(max(min_nm{0} - distance(g1,g2), 0)^order{0})".format(
                _name
            ),
            "params": {
                f"restr_k{_name}": force_config.restr_k[0],
                f"min_nm{_name}": force_config.min_nm[0],
                f"order{_name}": force_config.get("order", [2])[0],
            },
            "is_periodic": force_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info_min)

    def generate_dist_max(force_config):
        force_config.max_nm=[x * unit.nanometer for x in force_config.max_nm]
        info_max = {
            "grps": [force_config.grp1, force_config.grp2],
            "func": "(restr_k{0}/2)*(max(distance(g1,g2) - max_nm{0}, 0)^order{0})".format(
                _name
            ),
            "params": {
                f"restr_k{_name}": force_config.restr_k[0],
                f"max_nm{_name}": force_config.max_nm[0],
                f"order{_name}": force_config.get("order", [2])[0],
            },
            "is_periodic": force_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info_max)

    force_config.grp1 = force_config.grp1
    force_config.grp2 = force_config.grp2
    force_config.restr_k = [x * unit.kilojoules_per_mole for x in force_config.restr_k]

    return_ls = []
    _name = force_config.name
    if force_config.get("min_nm"):
        return_ls.append(generate_dist_min(force_config))
    if force_config.get("max_nm"):
        return_ls.append(generate_dist_max(force_config))

    return return_ls, get_update_params(
        force_config, ["min_nm", "max_nm", "restr_k", "order"]
    )


def generate_restraint_angle(force_config):
    # restraint the angle between three groups of atoms
    # if min_degree is defined, add a restraint when angle is smaller than min_degree
    # if max_degree is defined, add a restraint when angle is larger than max_degree
    def generate_angle_min(force_config):
        _name = force_config.name
        force_config.min_degree=[x * unit.degree for x in force_config.min_degree]
        info = {
            "grps": [
                force_config.grp1,
                force_config.grp2,
                force_config.grp3,
            ],
            "func": "(restr_k{0}/2)*(max(min_degree{0} - angle(g1, g2, g3), 0)^order{0})".format(
                _name
            ),
            "params": {
                f"restr_k{_name}": force_config.restr_k[0],
                f"min_degree{_name}": force_config.min_degree[0],
                f"order{_name}": force_config.get("order", [2])[0],
            },
            "is_periodic": force_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_angle_max(force_config):
        _name = force_config.name
        force_config.max_degree=[x * unit.degree for x in force_config.max_degree]
        info = {
            "grps": [
                force_config.grp1,
                force_config.grp2,
                force_config.grp3,
            ],
            "func": "(restr_k{0}/2)*(max(angle(g1, g2, g3) - max_degree{0}, 0)^order{0})".format(
                _name
            ),
            "params": {
                f"restr_k{_name}": force_config.restr_k[0],
                f"max_degree{_name}": force_config.max_degree[0],
                f"order{_name}": force_config.get("order", [2])[0],
            },
            "is_periodic": force_config.get("is_periodic", True),
        }
        return generate_CustomCentroidBondForce(info)

    force_config.grp1 = force_config.grp1
    force_config.grp2 = force_config.grp2
    force_config.grp3 = force_config.grp3
    force_config.restr_k = [x * unit.kilojoules_per_mole for x in force_config.restr_k]

    return_ls = []
    if force_config.get("min_degree"):
        return_ls.append(generate_angle_min(force_config))
    if force_config.get("max_degree"):
        return_ls.append(generate_angle_max(force_config))
    return return_ls, get_update_params(
        force_config, ["min_degree", "max_degree", "restr_k", "order"]
    )


def generate_restraint_dihedral(force_config):
    # restraint the dihedral angle between four groups of atoms
    # min_degree and max_degree are always needed because the dihedral angle is periodic
    # def fix_max_angle(min_angle, max_angle):
    #     # make max_angle always in the range of [min_angle, min_angle + 360)
    #     import math

    #     max_angle += 360 * math.ceil((min_angle - max_angle) / 360)
    #     return max_angle

    _name = force_config.name
    # force_config.max_degree = fix_max_angle(
    #     force_config.min_degree, force_config.max_degree
    # )
    force_config.grp1=force_config.grp1
    force_config.grp2=force_config.grp2
    force_config.grp3=force_config.grp3
    force_config.grp4=force_config.grp4
    force_config.restr_k = [x * unit.kilojoules_per_mole for x in force_config.restr_k]
    force_config.min_degree = [x * unit.degree for x in force_config.min_degree]
    force_config.max_degree=[x * unit.degree for x in force_config.max_degree]

    arctan_x = f"atan(tan((dihedral(g1,g2,g3,g4)-(min_degree{_name}+max_degree{_name})/2)/2))"
    arctan_half_diff = f"atan(tan((max_degree{_name} - min_degree{_name})/4))"
    energy_min = f"abs(min({arctan_x} - (-({arctan_half_diff})), 0))"
    energy_max = f"abs(max({arctan_x} - {arctan_half_diff}, 0))"
    info = {
        "grps": [
            force_config.grp1,
            force_config.grp2,
            force_config.grp3,
            force_config.grp4,
        ],
        "func": f"restr_k{_name}*({energy_min}+{energy_max})^order{_name}",
        "params": {
            f"restr_k{_name}": force_config.restr_k[0],
            f"min_degree{_name}": force_config.min_degree[0],
            f"max_degree{_name}": force_config.max_degree[0],
            f"order{_name}": force_config.get("order", [2])[0],
        },
        "is_periodic": force_config.get("is_periodic", True),
    }
    return generate_CustomCentroidBondForce(info), get_update_params(
        force_config, ["min_degree", "max_degree", "restr_k", "order"]
    )


# need system when add virtual particle
def generate_dist_ref_position(force_config):
    # restraint the distance between a group of atoms and a reference position
    # if min_nm is defined, add a restraint when distance is smaller than min_nm
    # if max_nm is defined, add a restraint when distance is larger than max_nm
    def generate_ref_position_min_restraint(force_config):
        force_config.min_nm = [x * unit.nanometer for x in force_config.min_nm]
        info = {
            "grps": [force_config.restr_grp],
            "func": "0.5*restr_k{0}*min(((x1-ref_x_nm{0})^2+(y1-ref_y_nm{0})^2+(z1-ref_z_nm{0})^2)^0.5-min_nm{0},0)^order{0}".format(
                _name
            ),
            "params": {
                f"restr_k{_name}": force_config.restr_k[0],
                f"ref_x_nm{_name}": force_config.ref_x_nm[0],
                f"ref_y_nm{_name}": force_config.ref_y_nm[0],
                f"ref_z_nm{_name}": force_config.ref_z_nm[0],
                f"min_nm{_name}": force_config.min_nm[0],
                f"order{_name}": force_config.get("order", [2])[0],
            },
            "is_periodic": force_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

    def generate_ref_position_max_restraint(force_config):
        force_config.max_nm = [x * unit.nanometer for x in force_config.max_nm]
        info = {
            "grps": [force_config.restr_grp],
            "func": "0.5*restr_k{0}*max(((x1-ref_x_nm{0})^2+(y1-ref_y_nm{0})^2+(z1-ref_z_nm{0})^2)^0.5-max_nm{0},0)^order{0}".format(
                _name
            ),
            "params": {
                f"restr_k{_name}": force_config.restr_k[0],
                f"ref_x_nm{_name}": force_config.ref_x_nm[0],
                f"ref_y_nm{_name}": force_config.ref_y_nm[0],
                f"ref_z_nm{_name}": force_config.ref_z_nm[0],
                f"max_nm{_name}": force_config.max_nm[0],
                f"order{_name}": force_config.get("order", [2])[0],
            },
            "is_periodic": force_config.get("is_periodic", False),
        }
        return generate_CustomCentroidBondForce(info)

    force_config.restr_grp = force_config.restr_grp
    if force_config.get("restr_k_per_atom"):
        force_config.restr_k = [
            x * unit.kilojoules_per_mole * len(force_config.restr_grp)
            for x in force_config.restr_k
        ]
    else:
        force_config.restr_k = [
            x * unit.kilojoules_per_mole for x in force_config.restr_k
        ]

    _name = force_config.name
    force_config.ref_x_nm = [x * unit.nanometer for x in force_config.ref_x_nm]
    force_config.ref_y_nm = [x * unit.nanometer for x in force_config.ref_y_nm]
    force_config.ref_z_nm = [x * unit.nanometer for x in force_config.ref_z_nm]

    return_ls = []
    if force_config.get("min_nm"):
        return_ls.append(generate_ref_position_min_restraint(force_config))
    if force_config.get("max_nm"):
        return_ls.append(generate_ref_position_max_restraint(force_config))

    return return_ls, get_update_params(
        force_config,
        [
            "ref_x_nm","ref_y_nm","ref_z_nm",
            "min_nm","max_nm",
            "restr_k","order",
        ],
    )


def generate_restraint_rmsd(force_config):
    import openmm
    from openmm.app import PDBxFile, PDBFile

    if force_config.ref_pos_file.endswith('.pdbx'):
        pos=PDBxFile(force_config.ref_pos_file).positions
    elif force_config.ref_pos_file.endswith('.pdb'): 
        pos=PDBFile(force_config.ref_pos_file).positions
    else:
        raise ValueError(f'ref_pos_file should be pdb or pdbx, {force_config.ref_pos_file} is not either')

    rmsd_cv = openmm.RMSDForce(pos,
                               force_config.restr_grp)
    _name=force_config.name
    force_config.restr_k = [
        x * unit.kilojoules_per_mole * len(force_config.restr_grp)
        for x in force_config.restr_k
    ]
    force_config.maxRMSD_nm = [x * unit.nanometer for x in force_config.maxRMSD_nm ]

    energy_expression = "(restr_k{0}/2)*max(0, RMSD-maxRMSD_nm{0})^2".format(_name)
    smd_force = openmm.CustomCVForce(energy_expression)
    smd_force.addCollectiveVariable('RMSD', rmsd_cv)
    smd_force.addGlobalParameter('maxRMSD_nm{0}'.format(_name), force_config.maxRMSD_nm[0])
    smd_force.addGlobalParameter("restr_k{0}".format(_name), force_config.restr_k[0])
    force_config['_force']=smd_force
    return smd_force, get_update_params(
        force_config,
        [
            "maxRMSD_nm",
            "restr_k",
        ],
    )
