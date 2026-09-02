from box import Box
import argparse

from openmm.app import PDBFile,PDBxFile
from openmm import XmlSerializer
import pandas as pd
import csv
def ats_info2_indices(ats_info, topology_info):
    indices = []
    for at_info in ats_info:
        indices.append(topology_info[at_info])
    return indices

def reorder_indices(indices):    
    if indices[0] > indices[-1]:
        indices.reverse()
    return indices
def parse_bonds(topology_info, system, parse_info):
    for force in system.getForces():
        if force.__class__.__name__ == 'HarmonicBondForce':
            break
    bond_params = {}
    for index in range(force.getNumBonds()):
        particle1, particle2, length, k = force.getBondParameters(index)
        force_indices = reorder_indices([particle1, particle2])
        for at_ls in parse_info["atoms"]:
            require_indices = reorder_indices(ats_info2_indices(at_ls, topology_info))
            if force_indices != require_indices:
                continue
            require_name = f"{at_ls[0]} - {at_ls[1]}"
            unit_length = length.unit
            unit_k = k.unit
            if require_name not in bond_params.keys():
                bond_params[require_name] = {
                    f"length {unit_length.get_symbol()}": [
                        length.value_in_unit(unit_length)
                    ],
                    f"k {unit_k.get_symbol()}": [k.value_in_unit(unit_k)],
                }
            else:
                bond_params[require_name][f"length {unit_length.get_symbol()}"].append(
                    length.value_in_unit(unit_length)
                )
                bond_params[require_name][f"k {unit_k.get_symbol()}"].append(
                    k.value_in_unit(unit_k)
                )
    parse_df = pd.DataFrame.from_dict(bond_params).T
    if parse_info.get('out_csv'):
        parse_df.to_csv(parse_info['out_csv'],
                        quoting=csv.QUOTE_ALL)
    else:
        print(parse_df)

def parse_angles(topology_info, system, parse_info):
    for force in system.getForces():
        if force.__class__.__name__ == 'HarmonicAngleForce':
            break
    angle_params = {}
    for index in range(force.getNumAngles()):
        particle1, particle2, particle3, angle, k = force.getAngleParameters(index)
        force_indices = reorder_indices([particle1, particle2, particle3])
        for at_ls in parse_info["atoms"]:
            require_indices = reorder_indices(ats_info2_indices(at_ls, topology_info))
            if force_indices != require_indices:
                continue
            require_name = f"{at_ls[0]} - {at_ls[1]} - {at_ls[2]}"
            unit_angle = angle.unit
            unit_k = k.unit
            if require_name not in angle_params.keys():
                angle_params[require_name] = {
                    f"angle {unit_angle.get_symbol()}": [
                        angle.value_in_unit(unit_angle)
                    ],
                    f"k {unit_k.get_symbol()}": [k.value_in_unit(unit_k)],
                }
            else:
                angle_params[require_name][f"angle {unit_angle.get_symbol()}"].append(
                    angle.value_in_unit(unit_angle)
                )
                angle_params[require_name][f"k {unit_k.get_symbol()}"].append(
                    k.value_in_unit(unit_k)
                )
    parse_df = pd.DataFrame.from_dict(angle_params).T
    if parse_info.get('out_csv'):
        parse_df.to_csv(parse_info['out_csv'],
                        quoting=csv.QUOTE_ALL)
    else:
        print(parse_df)

def parse_dihedrals(topology_info, system, parse_info):
    for force in system.getForces():
        if force.__class__.__name__ == 'PeriodicTorsionForce':
            break
    dihedral_params = {}
    for index in range(force.getNumTorsions()):
        particle1, particle2, particle3, particle4, periodicity, phase, k = force.getTorsionParameters(index)
        force_indices = reorder_indices([particle1, particle2, particle3, particle4])
        for at_ls in parse_info["atoms"]:
            require_indices = reorder_indices(ats_info2_indices(at_ls, topology_info))
            if force_indices != require_indices:
                continue
            require_name = f"{at_ls[0]} - {at_ls[1]} - {at_ls[2]} - {at_ls[3]}"
            unit_phase = phase.unit
            unit_k = k.unit
            if require_name not in dihedral_params.keys():
                dihedral_params[require_name] = {
                    f"phase {unit_phase.get_symbol()}": [
                        phase.value_in_unit(unit_phase)
                    ],
                    f"k {unit_k.get_symbol()}": [k.value_in_unit(unit_k)],
                    'periodicity': [periodicity]
                }
            else:
                dihedral_params[require_name][f"phase {unit_phase.get_symbol()}"].append(
                    phase.value_in_unit(unit_phase)
                )
                dihedral_params[require_name][f"k {unit_k.get_symbol()}"].append(
                    k.value_in_unit(unit_k)
                )
                dihedral_params[require_name]['periodicity'].append(periodicity)
    parse_df = pd.DataFrame.from_dict(dihedral_params).T
    if parse_info.get('out_csv'):
        parse_df.to_csv(parse_info['out_csv'],
                        quoting=csv.QUOTE_ALL)
    else:
        print(parse_df)

def parse_nonbonds(topology_info, system, parse_info):
    for force in system.getForces():
        if force.__class__.__name__ == 'NonbondedForce':
            break
    nonbonds_params = {}
    for require_name in parse_info["atoms"]:
        require_index = ats_info2_indices([require_name], topology_info)[0]
        chg, sigma , epsilon = force.getParticleParameters(require_index)
        unit_chg = chg.unit
        unit_sigma = sigma.unit
        unit_epsilon = epsilon.unit
        if require_name not in nonbonds_params.keys():
            nonbonds_params[require_name] = {
                f"charge {unit_chg.get_symbol()}": [
                    chg.value_in_unit(unit_chg)
                ],
                f"sigma {unit_sigma.get_symbol()}": [
                    sigma.value_in_unit(unit_sigma)
                ],
                f"epsilon {unit_epsilon.get_symbol()}": [
                    epsilon.value_in_unit(unit_epsilon)
                ],
            }
        else:
            nonbonds_params[require_name][f"charge {unit_chg.get_symbol()}"].append(
                chg.value_in_unit(unit_chg)
            )
            nonbonds_params[require_name][f"sigma {unit_sigma.get_symbol()}"].append(
                sigma.value_in_unit(unit_sigma)
            )
            nonbonds_params[require_name][f"epsilon {unit_epsilon.get_symbol()}"].append(
                epsilon.value_in_unit(unit_epsilon)
            )
    parse_df = pd.DataFrame.from_dict(nonbonds_params).T
    if parse_info.get('out_csv'):
        parse_df.to_csv(parse_info['out_csv'],
                        quoting=csv.QUOTE_ALL)
    else:
        print(parse_df)

def get_topology_info(topology):
    """
    Get topology information from the topology object.
    """
    topology_info = {}
    for atom in topology.atoms():
        chain_id=atom.residue.chain.id
        resid=atom.residue.id
        name=atom.name
        topology_info[f'{chain_id} {resid} {name}'] = atom.index
        
    return topology_info
def main(args):
    config = Box.from_yaml(filename=args.config)
    structure_f=config.input_files.complex
    if structure_f.endswith('.pdb'):
        topology = PDBFile(structure_f).topology
    elif structure_f.endswith('.pdbx') or structure_f.endswith('.cif'):
        topology = PDBxFile(structure_f).topology
    else:
        raise ValueError("Unsupported file format: {}".format(structure_f))
    topology_info = get_topology_info(topology)
    system = XmlSerializer.deserialize(open(config.input_files.system,"r").read())
    for force_type,parse_info in config['parse_params'].items():
        if force_type == 'bonds':
            parse_bonds(topology_info, system, parse_info)
        elif force_type == 'angles':
            parse_angles(topology_info, system, parse_info)
        elif force_type == 'dihedrals':
            parse_dihedrals(topology_info, system, parse_info)
        elif force_type == 'nonbonds':
            parse_nonbonds(topology_info, system, parse_info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="force field parameter parser")
    parser.add_argument("config", type=str, help="configuration file")
    args = parser.parse_args()
    main(args)
