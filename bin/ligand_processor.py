import argparse
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS,AllChem,rdDetermineBonds,rdMolTransforms
def convert_format(args):
    """转换分子文件格式"""
    # 读取输入文件
    if args.input.endswith('.sdf'):
        mol = Chem.MolFromMolFile(args.input,removeHs=False)
    elif args.input.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(args.input,removeHs=False)
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

def add_chirals_constraint(chirals,match_ls,ff,degree_tolerance=5):
    for _id,chiral in chirals.items():
        if len(set(match_ls) & set(chiral['neighbors']))>2: continue 
        for id_str,dih_deg in chiral['dihedrals'].items():
            at1,at2,at3,at4=[int(x) for x in id_str.split('-')]
            ff.MMFFAddTorsionConstraint(at1,at2,at3,at4,
                                        False,
                                        dih_deg-degree_tolerance,
                                        dih_deg+degree_tolerance,
                                        1.e3)

def get_chiral_dihedrals(mol,chiral_id,match_ls,confid=0):
    neighbors_id = [n.GetIdx() for n in mol.GetAtomWithIdx(chiral_id).GetNeighbors()]
    assert len(neighbors_id) == 4
    at1,at2,at3,at4 = neighbors_id
    conf = mol.GetConformer(confid)
    dihs={}
    for at1 in neighbors_id:
        if at1 not in match_ls: continue
        for at3 in neighbors_id:
            if at3 not in match_ls or at3 in [at1]: continue
            for at4 in neighbors_id:
                if at4 in match_ls or at4 in [at1,at3]: continue
                dihs[f'{at1}-{chiral_id}-{at3}-{at4}']= rdMolTransforms.GetDihedralDeg(conf, at1,chiral_id,at3,at4)
    return dihs

def get_chirals(mol,match_ls):
    """获取分子中所有手性碳的信息"""
    chiral_centers = {}
    for atom in mol.GetAtoms():
        if atom.HasProp("_ChiralityPossible") and atom.HasProp("_CIPCode"):
            if atom.GetProp("_CIPCode") in ["R", "S"]:
                chiral_centers[atom.GetIdx()]={
                    "cip_code": atom.GetProp("_CIPCode"),
                    "neighbors": [n.GetIdx() for n in atom.GetNeighbors()],
                    "dihedrals": get_chiral_dihedrals(mol,atom.GetIdx(),match_ls)
                }
    return chiral_centers

def kabsch_align(mobile_coords, target_coords):
    """
    经典Kabsch算法：计算最优旋转矩阵+平移向量，实现刚体叠合（无畸变）
    :param mobile_coords: 分子1(待对齐)的匹配原子坐标 N×3 numpy数组
    :param target_coords: 分子2(参考)的匹配原子坐标 N×3 numpy数组
    :return: 平移向量t, 旋转矩阵R
    """
    # 步骤1：计算两个匹配原子集的几何中心（质心）
    centroid_mobile = np.mean(mobile_coords, axis=0)
    centroid_target = np.mean(target_coords, axis=0)
    
    # 步骤2：平移到原点（消除平移差异）
    mobile_centered = mobile_coords - centroid_mobile
    target_centered = target_coords - centroid_target

    # 步骤3：计算协方差矩阵，SVD分解求最优旋转矩阵（核心）
    H = np.dot(mobile_centered.T, target_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # 修正旋转矩阵的手性（避免镜像翻转）
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = np.dot(Vt.T, U.T)
    
    # 平移向量 = 参考分子质心 - 旋转后的移动分子质心
    t = centroid_target - np.dot(centroid_mobile, R)
    return R, t

def get_sliced_mol_pos(mol,ids):
    pos=[]
    for i in ids:
        pos.append(mol.GetConformer(0).GetAtomPosition(i))
    return pos

def align_mol_pos(mol_pos,mol_top,
                  match_pos,match_top,
                 ):
    mol_pos_match_coords = get_sliced_mol_pos(mol_pos, match_pos)
    mol_top_match_coords = get_sliced_mol_pos(mol_top, match_top)
    mol_top_coords = get_sliced_mol_pos(
        mol_top, list(range(mol_top.GetNumAtoms()))
    )
    R, t = kabsch_align(mol_top_match_coords, mol_pos_match_coords)
    mol_top_aligned_coords = np.dot(mol_top_coords, R) + t
    return mol_top_aligned_coords

def optimize_with_constraints(mol,constraints_ids,
                              chirals=None):
    mp = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
    for i in constraints_ids:
        ff.MMFFAddPositionConstraint(i, 0, 1.e4)
    if chirals is not None:
        add_chirals_constraint(chirals,constraints_ids,ff)    
    ff.Minimize(maxIts=1000000,
                    energyTol=1e-6)
    return mol
def mol_smiles_to_pos_mol(mol_pos,smiles,
                          ignore_pos_ids=[],
                          ignore_top_ids=[],
                          atom_compare=rdFMCS.AtomCompare.CompareElements,
                         bond_compare=rdFMCS.BondCompare.CompareAny,
                         ):
    mol_top=Chem.MolFromSmiles(smiles)
    mol_top = Chem.AddHs(mol_top)
    mols=[mol_pos,mol_top]
    params = rdFMCS.MCSParameters()
    params.AtomTyper = atom_compare
    params.BondTyper = bond_compare
    mcs = rdFMCS.FindMCS(mols, params)

    match_pos = mol_pos.GetSubstructMatch(mcs.queryMol)
    match_top = mol_top.GetSubstructMatch(mcs.queryMol)
    match_pos, match_top = zip(
        *[
            (x, y)
            for x, y in zip(match_pos, match_top)
            if x not in ignore_pos_ids and y not in ignore_top_ids
        ]
    )

    AllChem.EmbedMolecule(mol_top)
    original_chirals = get_chirals(mol_top,match_top)
    conf = mol_top.GetConformer(0)

    Chem.AllChem.AlignMol(
        mol_top,
        mol_pos,
        atomMap=list(zip(match_top, match_pos)),
    )

    mol_copy = Chem.Mol(mol_top)
    max_iter=100
    for iter in range(max_iter):
        Chem.AllChem.AlignMol(
            mol_top,
            mol_pos,
            atomMap=list(
                zip(match_top,match_pos)
            ),
        )
        for id1,id2 in zip(match_pos,match_top):
            _pos = mol_pos.GetConformer(0).GetAtomPosition(id1)
            conf.SetAtomPosition(id2, _pos)    

        mol_top = optimize_with_constraints(mol_top,
                                  match_top,
                                  chirals=original_chirals)

        mol_top = fix_CH_angle(mol_top)
        mol_top = optimize_with_constraints(
            mol_top, [at.GetIdx() for at in mol_top.GetAtoms() if at.GetSymbol() != "H"]
        )

        rms = Chem.AllChem.AlignMol(
            mol_copy,
            mol_top,
            atomMap=list(
                zip(
                    list(range(mol_top.GetNumAtoms())),
                    list(range(mol_top.GetNumAtoms())),
                )
            ),
        )
        if rms < 0.001:
            print(f'converge after {iter+1} iterations with rms: {rms}')
            break
        mol_copy = Chem.Mol(mol_top)

    print(f'pose change in the final iter: {rms}')
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

    return mol

def pdb_to_mol_custom_threshold(pdb_file, max_bond_length=1.8, sanitize=True):
    """基于自定义距离阈值从PDB生成分子"""
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
        for j in range(i+1, mol.GetNumAtoms()):                
            # 计算原子间距离
            dist = np.linalg.norm(
                conf.GetAtomPosition(i) - 
                conf.GetAtomPosition(j)
            )
            
            # 应用自定义阈值
            if dist > max_bond_length:
                _mol.RemoveBond(i, j)
    mol=_mol.GetMol() 
    # 3. 选择性执行化学检查
    if sanitize:
        Chem.SanitizeMol(mol)
    
    return mol


def pos_smiles2sdf(args):
    """将SMILES结构匹配到坐标文件并生成SDF"""
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
        ignore_pos_ids=[int(x) - 1 for x in args.ignore_pos_ids.split(",") if x != ""],
        bond_compare=rdFMCS.BondCompare.CompareAny,
    )

    if args.output.endswith('.sdf'):
        Chem.MolToMolFile(mol, args.output)
    elif args.output.endswith('.pdb'):
        Chem.MolToPDBFile(mol, args.output)
    elif args.output.endswith('.xyz'):
        Chem.MolToXYZFile(mol, args.output)
    else:
        raise ValueError(f"不支持输出格式: {args.output}")
    print(f"生成文件: {args.output}")


def conformer_generation(mol,N_CONF=100):
    # Generate conformers
    p = AllChem.ETKDGv2()
    p.verbose = True

    #Check if it's using the torsion angle parameters from the experimental database
    print(f'Use torsion angle parameters: {p.useExpTorsionAnglePrefs}')

    #p is for the generation method (here ETKDGv2 assigned as above)
    #The generation is stochastic
    cids = AllChem.EmbedMultipleConfs(mol, N_CONF, p)

    #double check the num. of conformers
    n_conf = mol.GetNumConformers()
    print(f'{n_conf} confs generated')

    #Optimize and calculate energy using a molecular mechanics force field.
    #Doing a simple calc. here just to pick cluster centers.
    #Note the converged molecules are local minimum, the results are related with start state
    cenergy = []
    print(f'minimizing {n_conf} confs...')
    for conf in cids:
        ITER_NUM=50
        converged = Chem.AllChem.MMFFOptimizeMolecule(mol,confId=conf,maxIters=ITER_NUM)
        while converged != 0:
            ITER_NUM += 50
            converged = Chem.AllChem.MMFFOptimizeMolecule(mol,confId=conf,maxIters=ITER_NUM)
        if converged != 0:
            print(f'rotamer{conf} not converged after {ITER_NUM} iteration')
        mp = Chem.AllChem.MMFFGetMoleculeProperties(mol)
        cenergy.append(Chem.AllChem.MMFFGetMoleculeForceField(mol,mp,confId=conf).CalcEnergy())
    
    sorted_cids = sorted(cids,key=lambda cid: cenergy[cid])
    print('Conformations all minimized!')
    return sorted_cids,cenergy

def smiles2sdf(args):
    """将SMILES结构匹配到坐标文件并生成SDF"""

    from rdkit.ML.Cluster import Butina

    smiles = args.smiles
    output_f = args.output
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    sorted_cids,cenergy = conformer_generation(mol,N_CONF=300)
    dmat = AllChem.GetConformerRMSMatrix(mol, prealigned=True)
    
    cluster_groups = Butina.ClusterData(dmat, mol.GetNumConformers(), 1, isDistData=True, reordering=False)
    print(f'{len(cluster_groups)} groups clustered from {mol.GetNumConformers()} confs!')
    w = Chem.SDWriter(output_f)
    w.write(mol,confId=cluster_groups[0][0])
    w.flush()
    w.close()
    print(f'sdf saved to: {output_f}')
    return mol

def reorder_sdf(args):
    input_f = args.input
    order_str = args.order
    order = [int(x)-1 for x in order_str.split(',')]

    mol = Chem.MolFromMolFile(input_f,removeHs=False)
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
    AllChem.EmbedMolecule(new_mol)
    conf = new_mol.GetConformer(0)
    for idx in range(new_mol.GetNumAtoms()):
        _pos = mol.GetConformer(0).GetAtomPosition(order[idx])
        conf.SetAtomPosition(idx, _pos)   
    Chem.SanitizeMol(new_mol)
    Chem.MolToMolFile(new_mol,
                    args.output)

def main():
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
    # parser_smiles.add_argument('--fix_CH', action='store_true', default=False,
    #                          help='修复CH键角问题')
    parser_smiles.add_argument('--ignore_pos_ids',default='',
                             help='pos文件中不用于匹配的atom id列表, 逗号分隔, 从1开始计数')
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
        "-od","--order",
        required=True,
        help='原子顺序, 逗号分隔的原子索引列表, 从1开始计数。\
如希望现在的atom 1,2,3 按2,3,1顺序排列, 则输入"2,3,1"',
    )
    parser_convert.add_argument('-o', '--output', required=True,
                              help='输出文件(.sdf)')
    parser_convert.set_defaults(func=reorder_sdf)

    parser_smiles2sdf = subparsers.add_parser('smiles2sdf', 
                                         help='将SMILES结构匹配到坐标文件并生成SDF')
    parser_smiles2sdf.add_argument('-s', '--smiles', required=True,
                             help='目标SMILES字符串')
    parser_smiles2sdf.add_argument('-o', '--output', required=True,
                              help='输出文件(.sdf/.pdb/.xyz)')
    parser_smiles2sdf.set_defaults(func=smiles2sdf)

    args = parser.parse_args()
    args.func(args)  # 调用对应的函数

if __name__ == '__main__':
    main()
