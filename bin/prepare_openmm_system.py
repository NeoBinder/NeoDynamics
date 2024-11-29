import argparse
import json
import os

import numpy as np
from box import Box
from openmm import XmlSerializer, app, unit

from neomd.builder.forcefiled import ComplexForceField
from neomd.builder.ligand import ligands_from_config
from neomd.io.system_loader import load_complex


def custom_bonds(top, pos, cutom_config):
    for resname, res_bonds in cutom_config.items():
        if resname not in top._standardBonds:
            bonds = []
            print(f"res {resname} not in top._standardBonds")
            top._standardBonds[resname] = bonds
        else:
            raise Exception(
                f"res {resname} found in top._standardBonds,cannot add bonds"
            )
        if res_bonds.get("bonds_from_ffxml"):
            import xml.etree.ElementTree as etree

            tree = etree.parse(res_bonds["bonds_from_ffxml"])
            find_res = 0
            if tree.getroot().find("Residues") is not None:
                for residue in tree.getroot().find("Residues").findall("Residue"):
                    if residue.attrib["name"] == resname:
                        find_res = 1
                        for bond in residue.findall("Bond"):
                            bonds.append(
                                (bond.attrib["atomName1"], bond.attrib["atomName2"])
                            )
                        break
            if not find_res:
                raise ValueError(
                    'Cannot find info of residue "{}" in file "{}"'.format(
                        resname, res_addH["H_from_ffxml"]
                    )
                )
        if res_bonds.get("cutsom_bonds"):
            for bond in res_bonds["cutsom_bonds"]:
                bonds.append((bond[0], bond[1]))
    top._bonds = []
    top.createStandardBonds()
    top.createDisulfideBonds(pos)


def custom_addH(modeller, forcefield, custom_config):
    infinity = float("Inf")
    for resname, res_addH in custom_config.items():
        data = modeller._ResidueData(resname)
        modeller._residueHydrogens[resname] = data
        if res_addH.get("H_from_ffxml"):
            import xml.etree.ElementTree as etree

            tree = etree.parse(res_addH["H_from_ffxml"])
            find_res = 0
            if tree.getroot().find("Residues") is not None:
                for residue in tree.getroot().find("Residues").findall("Residue"):
                    if residue.attrib["name"] == resname:
                        find_res = 1
                        H_parents = {}
                        for atom in residue.findall("Atom"):
                            at_type = atom.attrib["type"]
                            if (
                                forcefield.forcefield._atomTypes[at_type].element.symbol
                                == "H"
                            ):
                                H_parents[atom.attrib["name"]] = None
                        for bond in residue.findall("Bond"):
                            at1 = bond.attrib["atomName1"]
                            at2 = bond.attrib["atomName2"]
                            if at1 in H_parents:
                                H_parents[at1] = at2
                            elif at2 in H_parents:
                                H_parents[at2] = at1
                        break
            if not find_res:
                raise ValueError(
                    'Cannot find info of residue "{}" in file "{}"'.format(
                        resname, res_addH["H_from_ffxml"]
                    )
                )
            for hydrogen, parent in H_parents.items():
                maxph = infinity
                atomVariants = None
                terminal = None
                data.hydrogens.append(
                    modeller._Hydrogen(hydrogen, parent, maxph, atomVariants, terminal)
                )


def make_system(
    protein_config, ligands_config, forcefield_kwargs, sys_params, additional_config
):
    protein = None
    ligands = None
    box_vectors = None
    modeller = None

    # if system without protien,no box_vectos/modeller get, then get them from ligands
    if protein_config:
        protein = load_complex(protein_config.path)
        if protein_config.get("custom_res_bonds"):
            custom_bonds(
                protein.topology, protein.positions, protein_config["custom_res_bonds"]
            )
        box_vectors = protein.getTopology().getPeriodicBoxVectors()
        modeller = app.Modeller(protein.topology, protein.positions)
    if ligands_config is not None:
        ligands = ligands_from_config(ligands_config)

    # if didn't get box_vectors from protien,get it from ligands
    if box_vectors is None:
        from openmm.app.internal.unitcell import computePeriodicBoxVectors

        _unit = unit.nanometers
        pos_list = [
            _lig.molecule.conformers[0].value_in_unit(_unit) for _lig in ligands
        ]
        pos_np = np.concatenate(pos_list, axis=0) * _unit
        box_size = max(pos_np.max(axis=0) - pos_np.min(axis=0)) + 2 * 1 * _unit
        _size = box_size.value_in_unit(_unit)
        _angle = 90 * np.pi / 180.0
        box_vectors = computePeriodicBoxVectors(
            _size, _size, _size, _angle, _angle, _angle
        )

    forcefield = ComplexForceField(**forcefield_kwargs)
    if ligands:
        gaff_generator = forcefield.init_gaff_generator()
        for ligand in ligands:
            # give unique names for all atoms in the molecule, so that we don't need
            # to rename them when generate topology and template from molecule
            ligand.generate_unique_atom_names()
            ligand_mol = ligand.molecule
            if modeller is None:
                modeller = app.Modeller(
                    ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0]
                )
            else:
                modeller.add(
                    ligand_mol.to_topology().to_openmm(), ligand_mol.conformers[0]
                )
            _res = [res for res in modeller.topology.residues()][-1]
            forcefield.add_molecule(
                ligand.molecule,
                gaff_generator,
                residue=_res,
                template_path=ligand.template_path,
            )
        forcefield.forcefield.registerTemplateGenerator(gaff_generator.generator)

    if additional_config is None:
        additional_config = {}
    additional_config = {
        "add_hydrogens": True,
        "add_solv_ions": True,
        "ion_Strength": 0.1,
        **additional_config,
    }

    if protein_config:
        if protein_config.get("custom_res_addH"):
            custom_addH(modeller, forcefield, protein_config["custom_res_addH"])

    if additional_config.get("add_hydrogens"):
        res_ls = [res for res in modeller.topology.residues()]
        resname_ls = len(res_ls) * [None]
        if protein_config:
            if protein_config.get("custom_resname_dict"):
                for _resid_resname in protein_config.get("custom_resname_dict"):
                    for resid, resname in _resid_resname.items():
                        mismatch = 0
                        if (
                            resname in ["CYS", "CYX", "CYM"]
                            and res_ls[resid - 1].name != "CYS"
                        ):
                            mismatch = 1
                        elif (
                            resname in ["HID", "HIE", "HIP"]
                            and res_ls[resid - 1].name != "HIS"
                        ):
                            mismatch = 1
                        if mismatch:
                            raise ValueError(
                                f"residue {resid}:specified {resname} not is not valid name of residue {res_ls[resid-1].name}\n"
                            )
                        else:
                            resname_ls[resid - 1] = resname
        resvariant = modeller.addHydrogens(variants=resname_ls)
        for index, res in enumerate(modeller.topology.residues()):
            if resvariant[index] is None:
                continue
            print(
                "residue {} {} change form {} to {}\n".format(
                    res.id, res.chain.id, res.name, resvariant[index]
                )
            )
            res.name = resvariant[index]
    box_center_vec = 0.5 * box_vectors[0] + 0.5 * box_vectors[1] + 0.5 * box_vectors[2]
    move_vec = box_center_vec - modeller.positions.mean()
    for i in range(len(modeller.positions)):
        modeller.positions[i] += move_vec
    if additional_config.get("add_solv_ions"):
        modeller.addSolvent(
            forcefield,
            boxVectors=box_vectors,
            ionicStrength=additional_config.get("ion_Strength") * unit.molar,
        )
    sys_args = ComplexForceField.sys_params_from_config(sys_params)
    system = forcefield.createSystem(modeller.topology, **sys_args)
    return modeller, ligands, system


def prepare_system(config):
    config = Box.from_yaml(filename=args.config)
    if config.get("from_gromacs"):
        gro = app.GromacsGroFile(config["from_gromacs"].get("gro"))
        _top = app.GromacsTopFile(
            config["from_gromacs"].get("top"),
            periodicBoxVectors=gro.getPeriodicBoxVectors(),
            includeDir=config["from_gromacs"].get("ff_path"),
        )
        sys_args = ComplexForceField.sys_params_from_config(config.get("system_params"))
        system = _top.createSystem(**sys_args)
        top = _top.topology
        pos = gro.positions
        ligands = None
    elif config.get("from_amber"):
        coord = app.AmberInpcrdFile(config["from_amber"].get("inpcrd"))
        _top = app.AmberPrmtopFile(config["from_amber"].get("prmtop"))
        sys_args = ComplexForceField.sys_params_from_config(config.get("system_params"))
        system = _top.createSystem(**sys_args)
        top = _top.topology
        pos = coord.positions
        ligands = None
    else:
        modeller, ligands, system = make_system(
            config.get("protein"),
            config.get("ligands"),
            forcefield_kwargs=config.get("ff_setting", None),
            sys_params=config.get("system_params"),
            additional_config=config.get("additional"),
        )
        top = modeller.topology
        pos = modeller.positions
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    ligand_path = os.path.join(config.output_dir, "ligand.json")
    solv_path = os.path.join(config.output_dir, "solv.pdbx")

    app.PDBxFile.writeFile(
        top, pos, open(solv_path, "w"), keepIds=config.get("output_keepid", False)
    )
    if not ligands is None:
        with open(ligand_path, "w") as f:
            ligands_json = json.dumps(
                [json.loads(ligand.molecule.to_json()) for ligand in ligands]
            )
            f.write(ligands_json)

    system_path = os.path.join(config.output_dir, "system.xml")
    with open(system_path, "w") as f:
        print("\n *********system done***********\n")
        f.write(XmlSerializer.serialize(system))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare system handler")
    parser.add_argument("config", type=str, help="configuration file")
    args = parser.parse_args()
    prepare_system(args.config)
