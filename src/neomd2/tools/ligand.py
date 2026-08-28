"""Ligand workflow — verbatim port of v1 ``src/neomd/builder/ligand.py`` and
``bin/ligand_processor.py`` (v2 migration plan §5 item 2.6, §6 parity row
"Ligand processing").

Two halves, mirroring v1's split:

* **Ligand knowledge** (:class:`Ligand`, :func:`load_rdmol`,
  :func:`topology_from_rdkit`, :func:`ligands_from_config`) — what
  ``bin/prepare_openmm_system.py`` consumed through
  ``neomd.builder.ligand.ligands_from_config``: per-ligand ``{path, resname?,
  template_ffxml?, smiles?, partial_charges?}`` entries, SMILES graph
  validation (networkx isomorphism against the H-added SMILES target), and
  partial-charge assignment from a last-column-floats file.
  ``neomd2.system`` delegates here when a prepare config's ligand entries
  carry the ``smiles`` / ``partial_charges`` keys.
* **The ligand_processor CLI** (:func:`main` + one importable function per
  subcommand) — v1 ``bin/ligand_processor.py`` was a pure-RDKit structure
  utility with four subcommands:

  - ``convert``     — format conversion (.sdf/.pdb/.xyz; xyz input gets
    ``rdDetermineBonds.DetermineBonds``);
  - ``pos_smiles2sdf`` — map a SMILES onto an input coordinate file via MCS
    matching, embed + MMFF-minimize the SMILES topology with position and
    chiral-dihedral constraints (optionally ``--fix_CH``), write
    .sdf/.pdb/.xyz; PDB input either plain (``--sanitize default``) or
    distance-threshold bonded (``--sanitize distance --max_bond A``);
  - ``reorder_sdf`` — permute the atom order of an SDF (1-based comma list);
  - ``smiles2sdf``  — 300 ETKDGv2 conformers, per-conformer MMFF minimization
    to convergence, Butina clustering (threshold 1), write the cluster-center
    conformer of the lowest-energy cluster.

  v1's CLI shells out to NOTHING — every step is in-process RDKit (no
  antechamber/openbabel anywhere in it), so this port needs no
  :class:`~neomd2.tools.port.ToolRunner`; ligand *parameterization* (GAFF,
  antechamber) is the other tools module
  (:mod:`neomd2.tools.antechamber`), not this one.

Fidelity notes (deviations, all deliberate):

* ``main(argv=None)`` takes the argv (v1's ``main()`` read ``sys.argv``
  implicitly); the argparse surface — flags, defaults, subcommand names —
  is v1's verbatim, as is every user-facing message (including the Chinese
  ones) and the ValueError text
  ``"current smiles:{} \\t target smiles: {}"``.
* one v1 BUG is fixed in the port: ``reorder_sdf``'s function reads
  ``args.input`` but v1's parser never registered an ``-i/--input`` flag,
  so the subcommand crashed with ``AttributeError`` as shipped — the flag
  is added here (same spelling as every other subcommand).
* one rdkit-version adaptation in ``reorder_sdf``: the rebuilt molecule is
  sanitized BEFORE ``EmbedMolecule`` because current rdkit enforces the
  ``calcImplicitValence`` precondition there (v1's rdkit computed it
  lazily); the final molecule state and output are unchanged.
* :func:`Ligand.assign_partial_charges` keeps v1's exact unit semantics
  (values wrapped in an ``openmm`` elementary-charge ``Quantity`` before
  landing on the openff Molecule, then openff's private
  ``Molecule._normalize_partial_charges()`` makes the sum integral).
"""

from __future__ import annotations

import argparse
import os

import networkx as nx
import numpy as np
from openff.toolkit.topology import Molecule as openff_Molecule
from openmm import unit
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds, rdFMCS, rdMolTransforms

__all__ = [
    "Ligand",
    "load_rdmol",
    "topology_from_rdkit",
    "ligands_from_config",
    "convert_format",
    "pos_smiles2sdf",
    "smiles2sdf",
    "reorder_sdf",
    "main",
]


# ---------------------------------------------------------------------------
# Ligand knowledge (v1 src/neomd/builder/ligand.py)
# ---------------------------------------------------------------------------


def load_rdmol(ligand_path):
    """Load a molecule from a file using RDKit (v1 ``load_rdmol`` verbatim).

    Supports PDB, SDF, Mol2 and Mol inputs; an unrecognized suffix raises
    ``NotImplementedError("rdkit mol loading method not defined")`` — the v1
    message, kept verbatim (this is the builder/ligand.py loader, not
    system.py's ConfigValueError-flavored port of it).
    """
    if ligand_path.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(ligand_path, removeHs=False)
    elif ligand_path.endswith(".sdf"):
        supp = Chem.ForwardSDMolSupplier(ligand_path, removeHs=False)
        mol = next(supp)
    elif ligand_path.endswith(".mol2"):
        mol = Chem.MolFromMol2File(ligand_path, removeHs=False)
    elif ligand_path.endswith(".mol"):
        mol = Chem.MolFromMolFile(ligand_path, removeHs=False)
    else:
        raise NotImplementedError("rdkit mol loading method not defined")
    return mol


def topology_from_rdkit(rdkit_molecule):
    """The molecule's graph (atoms as nodes, bonds as edges) — v1 verbatim."""
    topology = nx.Graph()
    for atom in rdkit_molecule.GetAtoms():
        # Add the atoms as nodes
        topology.add_node(atom.GetIdx())

        # Add the bonds as edges
        for bonded in atom.GetNeighbors():
            topology.add_edge(atom.GetIdx(), bonded.GetIdx())

    return topology


class Ligand:
    """One ligand: an openff ``Molecule`` plus a template-path handle.

    v1 ``neomd/builder/ligand.py::Ligand`` verbatim (same attributes, same
    method names — consumers like v1 ``make_system`` called
    ``generate_unique_atom_names()`` on the wrapper and reached the openff
    molecule through ``.molecule``).
    """

    def __init__(self, molecule):
        self.molecule = molecule
        self.template_path = None

    @property
    def partial_charges(self):
        return self.molecule.partial_charges

    def to_rdkit(self):
        return self.molecule.to_rdkit()

    def assign_partial_charges(self, value, normalize=True):
        """Attach ``value`` (array-like floats, elementary charge) as the
        molecule's partial charges; when ``normalize``, openff's
        ``_normalize_partial_charges`` then makes the total integral — the
        v1 semantics verbatim, openmm ``Quantity`` wrapping included."""
        charges = unit.Quantity(value, unit.elementary_charge)
        self.molecule.partial_charges = charges
        if normalize:
            self.molecule._normalize_partial_charges()

    @classmethod
    def from_path(cls, ligand_path):
        """Load via :func:`load_rdmol` (Hs kept) into an openff Molecule;
        PDB inputs additionally get ``AssignAtomChiralTagsFromStructure``
        (v1 verbatim)."""
        rdkitmolh = load_rdmol(ligand_path)
        if os.path.splitext(ligand_path)[1] in [".pdb"]:
            Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)
        ligand_mol = openff_Molecule.from_rdkit(
            rdkitmolh, hydrogens_are_explicit=True)
        return Ligand(ligand_mol)

    def generate_unique_atom_names(self):
        self.molecule.generate_unique_atom_names(suffix='')


def ligands_from_config(config):
    """Build the :class:`Ligand` list from the v1 ``ligands`` config section.

    ``config`` maps ligand name -> ``{"path": ..., "resname"?: ...,
    "template_ffxml"?: ..., "smiles": ..., "partial_charges"?: path}``.
    v1 verbatim, including:

    * ``resname`` override, falling back to ``"LIG"`` when the loaded
      molecule's name is empty;
    * the SMILES graph check — the loaded molecule's networkx topology must
      be isomorphic to ``Chem.AddHs(MolFromSmiles(smiles))``'s, else
      ``ValueError("current smiles:{} \\t target smiles: {}")`` with the
      loaded molecule's canonical isomeric SMILES (message byte-identical);
    * ``partial_charges`` files parse as the LAST whitespace-separated float
      of each non-empty line, then go through
      :meth:`Ligand.assign_partial_charges`.
    """
    ligands = []
    for ligname, lig_info in config.items():
        ligand = Ligand.from_path(lig_info["path"])
        if lig_info.get("resname"):
            ligand.molecule.name = lig_info["resname"]
        elif ligand.molecule.name == "":
            ligand.molecule.name = "LIG"
        ligand.template_path = lig_info.get("template_ffxml")
        rdmol = ligand.to_rdkit()
        rdmol_top = topology_from_rdkit(rdmol)
        target_mol = Chem.MolFromSmiles(lig_info["smiles"])
        target_top = topology_from_rdkit(Chem.AddHs(target_mol))
        if not nx.is_isomorphic(rdmol_top, target_top):
            rdmol_smiles = Chem.MolToSmiles(rdmol, isomericSmiles=True,
                                            canonical=True)
            raise ValueError(
                "current smiles:{} \t target smiles: {}".format(
                    rdmol_smiles, lig_info["smiles"]
                )
            )
        if lig_info.get("partial_charges"):
            with open(lig_info["partial_charges"]) as f:
                charges = [float(line.strip().split()[-1]) for line in f
                           if line.strip()]
            charges = np.array(charges)
            ligand.assign_partial_charges(charges)
        ligands.append(ligand)
    return ligands


# ---------------------------------------------------------------------------
# ligand_processor (v1 bin/ligand_processor.py) — pure RDKit, no ToolRunner
# ---------------------------------------------------------------------------


def convert_format(args):
    """转换分子文件格式 (v1 ``convert_format`` verbatim)."""
    # 读取输入文件
    if args.input.endswith('.sdf'):
        mol = Chem.MolFromMolFile(args.input, removeHs=False)
    elif args.input.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(args.input, removeHs=False)
    elif args.input.endswith('.xyz'):
        mol = Chem.MolFromXYZFile(args.input)
        rdDetermineBonds.DetermineBonds(mol)
    else:
        raise ValueError(f"不支持输入格式: {args.input}")

    if mol is None:
        raise ValueError("无法读取输入文件或文件为空")

    # 写入输出文件
    if args.output.endswith('.sdf'):
        Chem.MolToMolFile(mol, args.output)
    elif args.output.endswith('.pdb'):
        Chem.MolToPDBFile(mol, args.output)
    elif args.output.endswith('.xyz'):
        Chem.MolToXYZFile(mol, args.output)
    else:
        raise ValueError(f"不支持输出格式: {args.output}")

    print(f"转换完成: {args.input} -> {args.output}")


def add_chirals_constraint(chirals, match_ls, ff, degree_tolerance=5):
    for _id, chiral in chirals.items():
        if len(set(match_ls) & set(chiral['neighbors'])) > 2:
            continue
        for id_str, dih_deg in chiral['dihedrals'].items():
            at1, at2, at3, at4 = [int(x) for x in id_str.split('-')]
            ff.MMFFAddTorsionConstraint(at1, at2, at3, at4,
                                        False,
                                        dih_deg - degree_tolerance,
                                        dih_deg + degree_tolerance,
                                        1.e4)


def get_chiral_dihedrals(mol, chiral_id, match_ls, confid=0):
    neighbors_id = [n.GetIdx() for n in
                    mol.GetAtomWithIdx(chiral_id).GetNeighbors()]
    assert len(neighbors_id) == 4
    conf = mol.GetConformer(confid)
    dihs = {}
    for at1 in neighbors_id:
        if at1 not in match_ls:
            continue
        for at3 in neighbors_id:
            if at3 not in match_ls or at3 in [at1]:
                continue
            for at4 in neighbors_id:
                if at4 in match_ls or at4 in [at1, at3]:
                    continue
                dihs[f'{at1}-{chiral_id}-{at3}-{at4}'] = \
                    rdMolTransforms.GetDihedralDeg(conf, at1, chiral_id, at3, at4)
    return dihs


def get_chirals(mol, match_ls):
    """获取分子中所有手性碳的信息"""
    chiral_centers = {}
    for atom in mol.GetAtoms():
        if atom.HasProp("_ChiralityPossible") and atom.HasProp("_CIPCode"):
            if atom.GetProp("_CIPCode") in ["R", "S"]:
                chiral_centers[atom.GetIdx()] = {
                    "cip_code": atom.GetProp("_CIPCode"),
                    "neighbors": [n.GetIdx() for n in atom.GetNeighbors()],
                    "dihedrals": get_chiral_dihedrals(mol, atom.GetIdx(),
                                                      match_ls)
                }
    return chiral_centers


def mol_smiles_to_pos_mol(mol_pos, smiles,
                          atom_compare=rdFMCS.AtomCompare.CompareElements,
                          bond_compare=rdFMCS.BondCompare.CompareAny,
                          ):
    """Embed the SMILES topology and pull it onto ``mol_pos`` coordinates via
    MCS matching, then MMFF-minimize with position (+ chiral dihedral)
    constraints (v1 verbatim)."""
    mol_top = Chem.MolFromSmiles(smiles)
    mol_top = Chem.AddHs(mol_top)
    mols = [mol_pos, mol_top]
    params = rdFMCS.MCSParameters()
    params.AtomTyper = atom_compare
    params.BondTyper = bond_compare
    mcs = rdFMCS.FindMCS(mols, params)

    match_pos = mol_pos.GetSubstructMatch(mcs.queryMol)
    match_top = mol_top.GetSubstructMatch(mcs.queryMol)

    AllChem.EmbedMolecule(mol_top)
    original_chirals = get_chirals(mol_top, match_top)
    conf = mol_top.GetConformer(0)
    for id1, id2 in zip(match_pos, match_top):
        _pos = mol_pos.GetConformer(0).GetAtomPosition(id1)
        conf.SetAtomPosition(id2, _pos)

    mp = AllChem.MMFFGetMoleculeProperties(mol_top)
    ff = AllChem.MMFFGetMoleculeForceField(mol_top, mp)
    for i in match_top:
        ff.MMFFAddPositionConstraint(i, 0, 1.e4)
    add_chirals_constraint(original_chirals, match_top, ff)
    ff.Minimize(maxIts=1000000)
    return mol_top


def calculate_angle(pos_h, pos_c, pos_a):
    """计算H-C-A键角(单位:度)"""
    # 将坐标转换为numpy数组
    vec_ch = np.array([pos_h.x - pos_c.x, pos_h.y - pos_c.y, pos_h.z - pos_c.z])
    vec_ca = np.array([pos_a.x - pos_c.x, pos_a.y - pos_c.y, pos_a.z - pos_c.z])

    # 计算点积和模长
    dot_product = np.dot(vec_ch, vec_ca)
    norm_ch = np.linalg.norm(vec_ch)
    norm_ca = np.linalg.norm(vec_ca)

    # 处理零向量（理论上不应出现）
    if norm_ch == 0 or norm_ca == 0:
        return 180.0

    cos_theta = dot_product / (norm_ch * norm_ca)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 避免数值误差
    return np.degrees(np.arccos(cos_theta))


def fix_CH_angle(mol):
    """Mirror any saturated-carbon hydrogen whose smallest H-C-A angle is
    under 90 degrees through the carbon, then UFF-relax with all non-H atoms
    fixed (v1 verbatim)."""
    conf = mol.GetConformer()

    # 遍历所有原子
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and atom.GetDegree() == 4:  # 找到饱和碳
            c_idx = atom.GetIdx()
            c_pos = conf.GetAtomPosition(c_idx)

            # 遍历碳的氢原子邻居
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'H':
                    h_idx = neighbor.GetIdx()
                    h_pos = conf.GetAtomPosition(h_idx)

                    # 计算与所有非H邻居的最小键角
                    min_angle = 180.0
                    has_valid_neighbor = False

                    for other in atom.GetNeighbors():
                        if other.GetSymbol() != 'H' and other.GetIdx() != h_idx:
                            a_idx = other.GetIdx()
                            a_pos = conf.GetAtomPosition(a_idx)
                            angle = calculate_angle(h_pos, c_pos, a_pos)
                            min_angle = min(min_angle, angle)
                            has_valid_neighbor = True

                    # 如果有有效邻居且最小角度小于90度，调整氢原子位置
                    if has_valid_neighbor and min_angle < 90:
                        new_x = 2 * c_pos.x - h_pos.x
                        new_y = 2 * c_pos.y - h_pos.y
                        new_z = 2 * c_pos.z - h_pos.z
                        conf.SetAtomPosition(h_idx, [new_x, new_y, new_z])

    # 创建力场并固定非氢原子
    ff = AllChem.UFFGetMoleculeForceField(mol)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'H':
            ff.AddFixedPoint(atom.GetIdx())

    # 执行优化（最多200次迭代）
    ff.Minimize(maxIts=1000)
    return mol


def pdb_to_mol_custom_threshold(pdb_file, max_bond_length=1.8, sanitize=True):
    """基于自定义距离阈值从PDB生成分子 (v1 verbatim: proximity-bonded PDB
    read, then every bond longer than ``max_bond_length`` removed)."""
    # 1. 禁用自动距离成键
    mol = Chem.MolFromPDBFile(
        pdb_file,
        sanitize=False,
        proximityBonding=True
    )
    if mol is None:
        return None

    # 2. 手动添加键（基于自定义阈值）
    conf = mol.GetConformer()
    _mol = Chem.rdchem.EditableMol(mol)
    for i in range(mol.GetNumAtoms()):
        for j in range(i + 1, mol.GetNumAtoms()):
            # 计算原子间距离
            dist = np.linalg.norm(
                conf.GetAtomPosition(i) -
                conf.GetAtomPosition(j)
            )

            # 应用自定义阈值
            if dist > max_bond_length:
                _mol.RemoveBond(i, j)
    mol = _mol.GetMol()
    # 3. 选择性执行化学检查
    if sanitize:
        Chem.SanitizeMol(mol)

    return mol


def pos_smiles2sdf(args):
    """将SMILES结构匹配到坐标文件并生成SDF (v1 ``pos_smiles2sdf``
    verbatim: input .pdb (``--sanitize default``) / .pdb via distance
    bonding (``--sanitize distance --max_bond``) / .sdf, MCS-matched onto
    ``--smiles``, optional ``--fix_CH``, output .sdf/.pdb/.xyz)."""
    if args.input.endswith('.pdb'):
        if args.sanitize == 'default':
            struct = Chem.MolFromPDBFile(args.input)
        elif args.sanitize == 'distance':
            struct = pdb_to_mol_custom_threshold(
                args.input,
                max_bond_length=args.max_bond
            )
    elif args.input.endswith('.sdf'):
        struct = Chem.MolFromMolFile(args.input)
    else:
        raise ValueError("输入文件格式不支持, 仅支持.pdb或.sdf文件")

    mol = mol_smiles_to_pos_mol(
        struct,
        args.smiles,
        bond_compare=rdFMCS.BondCompare.CompareAny
    )

    if args.fix_CH:
        mol = fix_CH_angle(mol)

    if args.output.endswith('.sdf'):
        Chem.MolToMolFile(mol, args.output)
    elif args.output.endswith('.pdb'):
        Chem.MolToPDBFile(mol, args.output)
    elif args.output.endswith('.xyz'):
        Chem.MolToXYZFile(mol, args.output)
    else:
        raise ValueError(f"不支持输出格式: {args.output}")
    print(f"生成文件: {args.output}")


def conformer_generation(mol, N_CONF=100):
    """ETKDGv2 conformer embedding + per-conformer MMFF minimization to
    convergence (v1 verbatim; returns cids sorted by energy)."""
    # Generate conformers
    p = AllChem.ETKDGv2()
    p.verbose = True

    # Check if it's using the torsion angle parameters from the experimental
    # database
    print(f'Use torsion angle parameters: {p.useExpTorsionAnglePrefs}')

    # p is for the generation method (here ETKDGv2 assigned as above)
    # The generation is stochastic
    cids = AllChem.EmbedMultipleConfs(mol, N_CONF, p)

    # double check the num. of conformers
    n_conf = mol.GetNumConformers()
    print(f'{n_conf} confs generated')

    # Optimize and calculate energy using a molecular mechanics force field.
    # Doing a simple calc. here just to pick cluster centers.
    # Note the converged molecules are local minimum, the results are related
    # with start state
    cenergy = []
    print(f'minimizing {n_conf} confs...')
    for conf in cids:
        ITER_NUM = 50
        converged = Chem.AllChem.MMFFOptimizeMolecule(mol, confId=conf,
                                                      maxIters=ITER_NUM)
        while converged != 0:
            ITER_NUM += 50
            converged = Chem.AllChem.MMFFOptimizeMolecule(mol, confId=conf,
                                                          maxIters=ITER_NUM)
        mp = Chem.AllChem.MMFFGetMoleculeProperties(mol)
        cenergy.append(
            Chem.AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf)
            .CalcEnergy())

    sorted_cids = sorted(cids, key=lambda cid: cenergy[cid])
    print('Conformations all minimized!')
    return sorted_cids, cenergy


def smiles2sdf(args):
    """SMILES -> single low-energy SDF conformer (v1 ``smiles2sdf``
    verbatim: 300 ETKDGv2 conformers, MMFF to convergence, Butina
    clustering at threshold 1, writes the first cluster's center)."""

    from rdkit.ML.Cluster import Butina

    smiles = args.smiles
    output_f = args.output
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    sorted_cids, cenergy = conformer_generation(mol, N_CONF=300)
    dmat = AllChem.GetConformerRMSMatrix(mol, prealigned=True)

    cluster_groups = Butina.ClusterData(dmat, mol.GetNumConformers(), 1,
                                        isDistData=True, reordering=False)
    print(f'{len(cluster_groups)} groups clustered from '
          f'{mol.GetNumConformers()} confs!')
    w = Chem.SDWriter(output_f)
    w.write(mol, confId=cluster_groups[0][0])
    w.flush()
    w.close()
    print(f'sdf saved to: {output_f}')
    return mol


def reorder_sdf(args):
    """Reorder an SDF's atoms by a 1-based comma-separated index list
    (v1 ``reorder_sdf`` verbatim)."""
    input_f = args.input
    order_str = args.order
    order = [int(x) - 1 for x in order_str.split(',')]

    mol = Chem.MolFromMolFile(input_f, removeHs=False)
    if mol is None:
        raise ValueError("无法读取输入文件或文件为空")

    if len(order) != mol.GetNumAtoms():
        raise ValueError("提供的原子顺序长度与分子原子数不匹配")

    # 创建新的分子并按指定顺序添加原子
    emol = Chem.EditableMol(Chem.Mol())
    for idx in order:
        atom = mol.GetAtomWithIdx(idx)
        emol.AddAtom(atom)

    # 添加键
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        new_begin_idx = order.index(begin_idx)
        new_end_idx = order.index(end_idx)
        emol.AddBond(new_begin_idx, new_end_idx, bond.GetBondType())

    new_mol = emol.GetMol()
    # sanitize before embedding: rdkit >= 2024 enforces the
    # calcImplicitValence precondition inside EmbedMolecule, which the
    # atoms copied through the EditableMol have not had run (v1's older
    # rdkit computed it lazily); the final state is identical
    Chem.SanitizeMol(new_mol)
    AllChem.EmbedMolecule(new_mol)
    conf = new_mol.GetConformer(0)
    for idx in range(new_mol.GetNumAtoms()):
        _pos = mol.GetConformer(0).GetAtomPosition(order[idx])
        conf.SetAtomPosition(idx, _pos)
    Chem.SanitizeMol(new_mol)
    Chem.MolToMolFile(new_mol, args.output)


def main(argv=None):
    """The v1 ligand_processor CLI (argparse surface verbatim); ``argv``
    defaults to ``sys.argv[1:]``."""
    parser = argparse.ArgumentParser(description='RDKit分子处理工具集')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # pos_smiles2sdf 命令
    parser_smiles = subparsers.add_parser('pos_smiles2sdf',
                                          help='将SMILES结构匹配到坐标文件')
    parser_smiles.add_argument('-i', '--input', required=True,
                               help='输入结构文件(.pdb/.sdf)')
    parser_smiles.add_argument('-s', '--smiles', required=True,
                               help='目标SMILES字符串')
    parser_smiles.add_argument('-o', '--output', required=True,
                               help='输出SDF文件路径')
    parser_smiles.add_argument('--sanitize', default='default',
                               choices=['default', 'distance'],
                               help='PDB处理方式')
    parser_smiles.add_argument('--max_bond', type=float, default=2,
                               help='距离成键阈值(Å)')
    parser_smiles.add_argument('--fix_CH', action='store_true', default=False,
                               help='修复CH键角问题')
    parser_smiles.set_defaults(func=pos_smiles2sdf)

    parser_convert = subparsers.add_parser('convert',
                                           help='转换分子文件格式')
    parser_convert.add_argument('-i', '--input', required=True,
                                help='输入文件(.sdf/.pdb/.xyz)')
    parser_convert.add_argument('-o', '--output', required=True,
                                help='输出文件(.sdf/.pdb/.xyz)')
    parser_convert.set_defaults(func=convert_format)

    parser_convert = subparsers.add_parser('reorder_sdf',
                                           help='转换sdf文件内原子顺序')
    parser_convert.add_argument('-i', '--input', required=True,
                                help='输入文件(.sdf)')
    parser_convert.add_argument(
        "-od", "--order",
        required=True,
        help='原子顺序, 逗号分隔的原子索引列表, 从1开始计数。\
如希望现在的atom 1,2,3 按2,3,1顺序排列, 则输入"2,3,1"',
    )
    parser_convert.add_argument('-o', '--output', required=True,
                                help='输出文件(.sdf)')
    parser_convert.set_defaults(func=reorder_sdf)

    parser_smiles2sdf = subparsers.add_parser('smiles2sdf',
                                              help='将SMILES结构匹配到坐标'
                                                   '文件并生成SDF')
    parser_smiles2sdf.add_argument('-s', '--smiles', required=True,
                                   help='目标SMILES字符串')
    parser_smiles2sdf.add_argument('-o', '--output', required=True,
                                   help='输出文件(.sdf/.pdb/.xyz)')
    parser_smiles2sdf.set_defaults(func=smiles2sdf)

    args = parser.parse_args(argv)
    args.func(args)  # 调用对应的函数


if __name__ == '__main__':  # pragma: no cover
    main()
