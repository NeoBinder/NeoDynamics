"""蛋白结构处理工具集 (基于 MDAnalysis)。

当前提供:
  - align     : 将 mobile 结构刚体叠合到 reference 结构上 (默认按蛋白 C-alpha 对齐),
                输出对齐后的完整结构 -- 保留配体/离子/水等所有原子, 它们随蛋白
                一起被同一个刚体变换移动。 (依赖 MDAnalysis)
  - align-seq : 当两条链序列不一致 (有缺失/插入/组氨酸变体) 时, 先用 BioPython
                PairwiseAligner 做双序列全局比对获取残基位置对应, 再按对应 CA
                计算刚体变换矩阵 apply 到 mobile 全结构。 (依赖 MDAnalysis + BioPython)
  - rmsd      : 计算两个结构在指定原子选择下的 RMSD (最优叠合后)。 (依赖 MDAnalysis)
  - select    : 按链/残基名/残基号/原子名等条件 (AND 组合) 筛选原子, 可叠加
                around (空间邻域) / point (球域) 条件, 返回命中原子的 index
                及属性信息。 (依赖 MDAnalysis)
  - replace   : 把 source PDB 中指定残基整块替换到 target PDB 的同位置 (文本级,
                不依赖 MDAnalysis)。原子行原样保留, 默认不重排序号, 默认丢弃 CONECT。
  - append    : 把 source PDB 中指定残基追加到 target PDB 末端, 保留源 chain id,
                自动补 TER。 (文本级, 不依赖 MDAnalysis)
  - setbox    : 改写/插入 CRYST1 盒子记录。 (文本级, 不依赖 MDAnalysis)
  - seq       : 从 PDB / mmCIF 坐标 (ATOM) 记录提取蛋白序列, 每条链一条
                fasta 记录; --chains 可限定只要哪些链。支持 .pdb / .cif /
                .mmcif 输入 (均为文本级解析, 不依赖 MDAnalysis; 不读 SEQRES)。
  - renum     : 每条 chain 从 start 开始连续重编 resid (A 链 1,2,3.., B 链 1,2,3..)。
                (文本级, 不依赖 MDAnalysis)

输入 PDB 可以同时包含蛋白、小分子配体、离子和水; 默认只取蛋白 C-alpha
参与叠合, 但可通过 --select 指定任意 MDAnalysis 选择语句 (例如对齐配体:
--select "resname LIG and name CA")。

注意: align/align-seq/rmsd/select 子命令依赖 MDAnalysis, 需用户自行安装
(不在项目 environment.yaml / pixi.toml 中声明); align-seq 额外依赖 BioPython。
replace/append/setbox 子命令为纯文本行级操作, 不依赖任何第三方库, 可独立使用。

用法示例:
  # 把 docked.pdb 叠合到 crystal.pdb, 输出 aligned.pdb
  python3 bin/protein_processor.py align -m docked.pdb -r crystal.pdb -o aligned.pdb

  # 用 backbone 对齐, 并启用质量加权
  python3 bin/protein_processor.py align -m docked.pdb -r crystal.pdb -o aligned.pdb \\
      --select "protein and backbone" --weights mass

  # 计算两结构的 C-alpha RMSD
  python3 bin/protein_processor.py rmsd -m docked.pdb -r crystal.pdb

  # 查询 A 链上 1-50 号残基的 C-alpha 原子 index
  python3 bin/protein_processor.py select -i protein.pdb --chain A --resid 1-50 --atom CA

  # 查询距离配体 LIG 5 Å 以内的蛋白重原子 (空间邻域)
  python3 bin/protein_processor.py select -i complex.pdb --around "resname LIG" 5.0 \\
      --protein --heavy

  # 只输出纯 index 列表 (便于 shell 管道), 0-based
  python3 bin/protein_processor.py select -i protein.pdb --resname ALA --format list

  # 两条链序列不一致时 (有缺失/插入/HIS 变体), 用序列比对做对齐
  python3 bin/protein_processor.py align-seq -m mobile.pdb -r ref.pdb -o aligned.pdb \\
      --chain A

  # 从多个 PDB / cif 的坐标记录提取蛋白序列, 每条链一条 fasta 记录
  python3 bin/protein_processor.py seq -i *_trunc.pdb -o all.fasta

  # 从 mmCIF 提取指定链的序列
  python3 bin/protein_processor.py seq -i 3htb.cif --chains A --id-format file+chain
"""
import argparse
import csv
import os
import sys

# 重依赖 (numpy / MDAnalysis) 为惰性导入: 仅 align/align-seq/rmsd/select 需要;
# 文本级子命令 (seq/replace/append/setbox) 可在无这些依赖的环境下独立使用。
np = None
mda = None
align = None
rms = None


def _require_mda():
    """惰性加载 numpy 与 MDAnalysis; 缺失时给出清晰报错。"""
    global np, mda, align, rms
    if mda is not None:
        return
    try:
        import numpy as _np
        import MDAnalysis as _mda
        from MDAnalysis.analysis import align as _align, rms as _rms
    except ImportError as e:
        raise ImportError(
            "该子命令依赖 numpy 与 MDAnalysis (项目未声明, 需用户自行安装): "
            f"{e}"
        )
    np, mda, align, rms = _np, _mda, _align, _rms


# 默认参与叠合/计算的原子选择: 蛋白 C-alpha
DEFAULT_SELECT = "protein and name CA"


def _resolve_weights(weights_arg):
    """将命令行 weights 参数转换为 alignto/rmsd 可接受的值。"""
    if weights_arg in (None, "none", "None"):
        return None
    return weights_arg  # 例如 "mass"


def _print_selection_info(mobile_ag, reference_ag):
    print(f"mobile    选择原子数: {mobile_ag.n_atoms}")
    print(f"reference 选择原子数: {reference_ag.n_atoms}")


def align_structures(args):
    """将 mobile 结构叠合到 reference 结构上并写出对齐后的完整结构。"""
    _require_mda()
    mobile = mda.Universe(args.mobile)
    reference = mda.Universe(args.reference)
    weights = _resolve_weights(args.weights)

    # 选择用于计算叠合变换的原子 (默认蛋白 C-alpha)
    if args.select_mobile or args.select_ref:
        if not (args.select_mobile and args.select_ref):
            raise ValueError(
                "需要同时提供 --select-mobile 和 --select-ref, 或都不提供以使用 --select"
            )
        select = (args.select_mobile, args.select_ref)
        mob_ag = mobile.select_atoms(args.select_mobile)
        ref_ag = reference.select_atoms(args.select_ref)
    else:
        select = args.select
        mob_ag = mobile.select_atoms(args.select)
        ref_ag = reference.select_atoms(args.select)
    _print_selection_info(mob_ag, ref_ag)
    if mob_ag.n_atoms == 0 or ref_ag.n_atoms == 0:
        raise ValueError("所选原子数为 0, 请检查 --select 选择条件")

    n_frames = len(mobile.trajectory)
    if args.all_frames and n_frames > 1:
        # 多帧 (如多 MODEL 的 PDB): 用 AlignTraj 对齐所有帧到 reference
        aligner = align.AlignTraj(
            mobile, reference, select=select,
            weights=weights, match_atoms=args.match_atoms, in_memory=True,
        )
        aligner.run()
        # 逐帧写出为多 MODEL 的 PDB
        with mda.Writer(args.output, n_atoms=mobile.atoms.n_atoms) as w:
            for ts in mobile.trajectory:
                w.write(mobile.atoms)
        print(f"已对齐 {n_frames} 帧并写入: {args.output}")
    else:
        # 单帧: alignto 直接变换整个 mobile Universe 的坐标 (含配体/离子/水)
        rmsd_before, rmsd_after = align.alignto(
            mobile, reference, select=select,
            weights=weights, match_atoms=args.match_atoms,
        )
        # 写出全部原子: 蛋白 + 配体 + 离子 + 水, 均已按叠合变换移动
        mobile.atoms.write(args.output)
        print(f"对齐前 RMSD: {rmsd_before:.4f} Å")
        print(f"对齐后 RMSD: {rmsd_after:.4f} Å")
        print(f"已写入: {args.output}")


# ---------------------------------------------------------------------------
# 基于序列比对的结构对齐 (align-seq)
#
# 当两条链序列不一致 (缺失/插入/HIS 变体 HID/HIE/HIN 等) 时, 直接按 CA 顺序
# 逐原子匹配会错位。这里先用 BioPython PairwiseAligner 做双序列全局比对
# (Needleman-Wunsch + BLOSUM62) 获取残基位置对应, 再用匹配的 CA 计算刚体
# 变换矩阵 (rotation_matrix), apply 到 mobile 全结构。
# ---------------------------------------------------------------------------

# 三字母 -> 一字母氨基酸映射 (含常见力场变体, 未知残基 -> 'X')
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # 组氨酸质子化/互变异构变体 -> H
    "HID": "H", "HIE": "H", "HIN": "H", "HIP": "H", "HSD": "H",
    "HSE": "H", "HSP": "H",
    # 胱氨酸/二硫键变体 -> C
    "CYX": "C", "CYM": "C",
    # 赖氨酸变体 (如 KCX 羧基化赖氨酸) -> K
    "KCX": "K", "LYP": "K", "LSN": "K",
    # 其它常见修饰, 归入最接近的标准残基
    "ASH": "D", "GLH": "E", "TYM": "Y", "S2P": "S", "SEP": "S",
    "TPO": "T", "PTR": "Y", "MLY": "K", "MSE": "M",
}


def _extract_chain_ca_sequence(universe, chain_id):
    """提取指定链的 CA 序列 (一字母) 及对应残基列表。

    返回 (seq_str, residues), 其中 residues 为按出现顺序的 MDAnalysis
    Residue 对象列表, 其索引与 seq_str 的字符一一对应。
    """
    ag = universe.select_atoms(f"segid {chain_id} and name CA")
    residues = list(ag.residues)
    # 去重 (select_atoms 可能因多原子残基返回重复 residue, 但 CA 每残基一个)
    seen = set()
    unique_res = []
    for res in residues:
        rid = res.resindex
        if rid not in seen:
            seen.add(rid)
            unique_res.append(res)
    residues = unique_res
    seq = "".join(_THREE_TO_ONE.get(res.resname.strip().upper(), "X")
                  for res in residues)
    return seq, residues


def _seq_pairwise_align(seq_mob, seq_ref, gap_open=-10.0, gap_extend=-0.5):
    """对两条序列做全局比对, 返回匹配残基对索引列表。

    返回 list[(mob_idx, ref_idx)], 索引为各自 ungapped 序列的 0-based 位置。
    gap 列 (任一侧为 '-') 不计入匹配。
    """
    from Bio.Align import PairwiseAligner, substitution_matrices

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = gap_open
    aligner.extend_gap_score = gap_extend

    alignments = aligner.align(seq_mob, seq_ref)
    aln = alignments[0]
    gapped_mob = aln[0]
    gapped_ref = aln[1]

    pairs = []
    i = j = 0
    for cm, cr in zip(gapped_mob, gapped_ref):
        if cm != "-" and cr != "-":
            pairs.append((i, j))
        if cm != "-":
            i += 1
        if cr != "-":
            j += 1
    n_gaps = len(gapped_mob) - len(pairs)
    return pairs, n_gaps


def _apply_rigid_transform(mobile_universe, mob_atoms, ref_atoms, weights=None):
    """用 mob_atoms -> ref_atoms 的最优刚体变换变换 mobile_universe 全部原子。

    复刻 MDAnalysis align._fit_to 流程:
      X' = R @ (X - mob_com) + ref_com
    返回 (rmsd_before, rmsd_after):
      rmsd_before -- 居中但未旋转时两侧坐标的 RMSD (反映对齐前差异)
      rmsd_after  -- 旋转后 (最终对齐) 的 RMSD
    """
    mob_com = mob_atoms.center(weights)
    ref_com = ref_atoms.center(weights)

    mob_coords_centered = mob_atoms.positions - mob_com
    ref_coords_centered = ref_atoms.positions - ref_com

    # 对齐前 RMSD: 居中但未旋转
    diff_before = mob_coords_centered - ref_coords_centered
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        rmsd_before = float(np.sqrt(np.sum(w[:, None] * diff_before ** 2)))
    else:
        rmsd_before = float(np.sqrt(np.mean(np.sum(diff_before ** 2, axis=1))))

    # 求解最优旋转 R (rotation_matrix 返回的 rmsd 即对齐后值)
    R, _ = align.rotation_matrix(
        mob_coords_centered, ref_coords_centered, weights=weights
    )

    # apply 到 mobile 全结构: 平移 -> 旋转 -> 平移
    mobile_universe.atoms.translate(-mob_com)
    mobile_universe.atoms.rotate(R)
    mobile_universe.atoms.translate(ref_com)

    # 对齐后 RMSD (用居中+旋转后的 mobile 坐标 vs 居中 ref 坐标)
    mob_transformed = (R @ mob_coords_centered.T).T
    diff_after = mob_transformed - ref_coords_centered
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
        rmsd_after = float(np.sqrt(np.sum(w[:, None] * diff_after ** 2)))
    else:
        rmsd_after = float(np.sqrt(np.mean(np.sum(diff_after ** 2, axis=1))))
    return rmsd_before, rmsd_after


def align_by_sequence(args):
    """基于序列比对的对齐: 处理两条链序列不一致的情况。

    流程: 提取两侧 chain CA 序列 -> BioPython 双序列全局比对 -> 按匹配
    残基对摘出对应 CA -> rotation_matrix 计算刚体变换 -> apply 到 mobile
    全结构 -> 写出对齐后 PDB。可选提取指定 resname 的对齐后坐标存独立 PDB。
    """
    _require_mda()
    mobile = mda.Universe(args.mobile)
    reference = mda.Universe(args.reference)

    # 1. 提取两侧链的 CA 序列
    mob_seq, mob_residues = _extract_chain_ca_sequence(mobile, args.chain)
    ref_seq, ref_residues = _extract_chain_ca_sequence(reference, args.chain)
    print(f"mobile    链 {args.chain}: {len(mob_seq)} 残基", file=sys.stderr)
    print(f"reference 链 {args.chain}: {len(ref_seq)} 残基", file=sys.stderr)
    if not mob_seq or not ref_seq:
        raise ValueError(
            f"链 {args.chain} 的 CA 序列为空, 请检查链 id 或 PDB 内容"
        )

    # 2. 双序列全局比对
    pairs, n_gaps = _seq_pairwise_align(
        mob_seq, ref_seq,
        gap_open=args.gap_open, gap_extend=args.gap_extend,
    )
    print(f"序列比对: {len(pairs)} 对匹配残基, {n_gaps} 列含 gap",
          file=sys.stderr)
    if len(pairs) < 3:
        raise ValueError(
            f"匹配残基数过少 ({len(pairs)}), 无法可靠计算刚体变换"
        )

    # 3. 按匹配对摘出 CA 原子, 组装一一对应的 AtomGroup
    mob_ca_atoms = [mob_residues[mi].atoms.select_atoms("name CA")[0]
                    for mi, _ in pairs]
    ref_ca_atoms = [ref_residues[ri].atoms.select_atoms("name CA")[0]
                    for _, ri in pairs]
    mob_ag = mda.core.groups.AtomGroup(mob_ca_atoms)
    ref_ag = mda.core.groups.AtomGroup(ref_ca_atoms)

    # 4. 质量加权 (可选)
    weights = mob_ag.masses if args.weights == "mass" else None

    # 5. 计算刚体变换并 apply 到 mobile 全结构
    rmsd_before, rmsd_after = _apply_rigid_transform(
        mobile, mob_ag, ref_ag, weights=weights
    )
    print(f"对齐前 RMSD (CA, 匹配残基): {rmsd_before:.4f} Å", file=sys.stderr)
    print(f"对齐后 RMSD (CA, 匹配残基): {rmsd_after:.4f} Å", file=sys.stderr)

    # 6. 写出对齐后的 mobile 全结构
    mobile.atoms.write(args.output)
    print(f"已写入对齐后结构: {args.output}", file=sys.stderr)


def compute_rmsd(args):
    """计算两个结构在指定原子选择下的 RMSD (最优叠合后)。"""
    _require_mda()
    mobile = mda.Universe(args.mobile)
    reference = mda.Universe(args.reference)
    weights = _resolve_weights(args.weights)

    mob_ag = mobile.select_atoms(args.select)
    ref_ag = reference.select_atoms(args.select)
    _print_selection_info(mob_ag, ref_ag)
    if mob_ag.n_atoms != ref_ag.n_atoms:
        raise ValueError(
            f"mobile 与 reference 选择原子数不一致 "
            f"({mob_ag.n_atoms} vs {ref_ag.n_atoms}), 无法逐原子计算 RMSD"
        )

    w = mob_ag.masses if weights == "mass" else None
    rmsd_value = rms.rmsd(
        mob_ag.positions, ref_ag.positions,
        weights=w, center=True, superposition=True,
    )
    print(f"RMSD ({args.select}): {rmsd_value:.4f} Å")


def _range_list_to_selection(values, key):
    """把 '1,3,5-7' 形式的列表展开为 MDAnalysis 选择语句片段。

    例如 key="resid", values="1,3,5-7" -> "resid 1 3 5 6 7"。
    逗号分隔的每段可以是单值或 'start-end' 闭区间。
    """
    tokens = []
    for part in values.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                start, end = end, start
            tokens.extend(str(i) for i in range(start, end + 1))
        else:
            tokens.append(part)
    if not tokens:
        raise ValueError(f"--{key} 没有提供有效值")
    return f"{key} " + " ".join(tokens)


def build_selection_string(args):
    """根据 CLI 条件参数构建 MDAnalysis 选择语句。

    chain / resname / resid / atom(--atom) 以 AND 组合;
    --protein / --heavy / --no-h 作为便捷开关叠加;
    若用户提供 --raw, 则作为额外原始选择语句 AND 进去。
    返回最终的选择字符串。
    """
    parts = []
    if args.chain:
        # 链 id 可为逗号分隔的多值, 如 "A,B"
        chains = [c.strip() for c in args.chain.split(",") if c.strip()]
        parts.append("segid " + " ".join(chains))
    if args.resname:
        names = [n.strip() for n in args.resname.split(",") if n.strip()]
        parts.append("resname " + " ".join(names))
    if args.resid:
        parts.append(_range_list_to_selection(args.resid, "resid"))
    if args.atom:
        names = [n.strip() for n in args.atom.split(",") if n.strip()]
        parts.append("name " + " ".join(names))
    if args.element:
        els = [e.strip() for e in args.element.split(",") if e.strip()]
        parts.append("element " + " ".join(els))
    if args.protein:
        parts.append("protein")
    if args.nucleic:
        parts.append("nucleic")
    if args.heavy:
        parts.append("not name H*")
    if args.no_h:
        parts.append("not name H*")
    if args.raw:
        parts.append(f"({args.raw})")

    if not parts:
        # 没给任何条件: 默认全部原子
        return "all"
    return " and ".join(f"({p})" for p in parts)


def select_atoms_query(args):
    """按链/残基名/残基号/原子名等条件筛选原子, 返回 index 及属性。

    支持 around (空间邻域: 距某参考选择一定距离内) 与
    point (球域: 距某坐标点一定距离内) 两种空间条件, 与基础属性条件
    以 AND 组合。
    """
    _require_mda()
    universe = mda.Universe(args.input)

    # 1) 基础属性选择
    base_sel = build_selection_string(args)

    # 2) 叠加空间邻域条件
    if args.around is not None:
        ref_sel, cutoff_str = args.around
        cutoff = float(cutoff_str)
        base_sel = f"({base_sel}) and around {cutoff} ({ref_sel})"
    if args.point is not None:
        x, y, z, cutoff = args.point
        base_sel = f"({base_sel}) and point {cutoff} {x} {y} {z}"

    print(f"选择语句: {base_sel}", file=sys.stderr)
    ag = universe.select_atoms(base_sel)

    indices = ag.indices.tolist()
    print(f"命中原子数: {len(indices)}", file=sys.stderr)

    if args.format == "list":
        # 纯 index 列表, 每行一个, 便于 shell 管道处理
        _output = args.output
        lines = [str(i) for i in indices]
        _write_lines(lines, _output)
        return

    # table / csv: 逐原子输出属性
    fieldnames = ["index", "segid", "chainid", "resid", "resname",
                  "atom_name", "element", "type", "x", "y", "z"]
    rows = []
    for atom in ag.atoms:
        pos = atom.position
        rows.append({
            "index": int(atom.index),
            "segid": atom.segid.strip(),
            "chainid": atom.segid.strip(),  # PDB 中 segid 常用于存 chain
            "resid": int(atom.resid),
            "resname": atom.resname.strip(),
            "atom_name": atom.name.strip(),
            "element": str(atom.element),
            "type": atom.type.strip(),
            "x": f"{pos[0]:.3f}",
            "y": f"{pos[1]:.3f}",
            "z": f"{pos[2]:.3f}",
        })

    if args.format == "csv":
        _output_csv(rows, fieldnames, args.output)
    else:  # table
        _output_table(rows, fieldnames, args.output)


def _write_lines(lines, output):
    """将纯文本行写出, output 为 None 时打印到 stdout。"""
    text = "\n".join(lines)
    if output:
        with open(output, "w") as f:
            f.write(text + "\n")
        print(f"已写入 {len(lines)} 个 index 到: {output}", file=sys.stderr)
    else:
        if text:
            print(text)


def _output_table(rows, fieldnames, output):
    """以对齐的文本表格形式输出。"""
    # 计算每列宽度
    str_rows = [[str(r[f]) for f in fieldnames] for r in rows]
    widths = [len(f) for f in fieldnames]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    header = fmt_row(fieldnames)
    sep = fmt_row(["-" * w for w in widths])
    body = "\n".join(fmt_row(r) for r in str_rows)
    text = f"{header}\n{sep}\n{body}" if rows else f"{header}\n{sep}"

    if output:
        with open(output, "w") as f:
            f.write(text + "\n")
        print(f"已写入 {len(rows)} 行到: {output}", file=sys.stderr)
    else:
        print(text)


def _output_csv(rows, fieldnames, output):
    """以 CSV 形式输出。"""
    if output:
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"已写入 {len(rows)} 行到: {output}", file=sys.stderr)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# 文本级 PDB 残基操作 (replace / append / setbox)
#
# 这组函数按 PDB 定长列解析原子记录 (ATOM/HETATM), 不调用 MDAnalysis, 不做
# 拓扑解析, 目的是"原样搬运文本行", 保证 round-trip 不被库工具重新格式化。
# 支持的列 (1-based):
#   1-6   记录名 (ATOM/HETATM)
#   7-11  原子序号 serial
#   13-16 原子名 name
#   18-20 残基名 resName
#   22    链 ID chainID
#   23-26 残基号 resSeq
#   77-78 元素 element
# ---------------------------------------------------------------------------

def _is_atom(line):
    """判断是否为原子记录行 (ATOM 或 HETATM)。"""
    return line.startswith("ATOM") or line.startswith("HETATM")


def _residue_key(line):
    """从原子行提取残基定位键 (chainID, resSeq, resName)。

    resSeq 保留为字符串 (含可能的插入码 iCode 在 27 列, 这里一并取 23-26);
    resName 取 18-20 列原样 (含前导/尾随空格), 便于精确匹配 ' OH' 这类残基。
    """
    return (line[21], line[22:26], line[17:20])


def _read_pdb(path):
    """读取 PDB 文件, 返回 (atoms, header_lines, cryst1_line)。

    atoms        : list[str]  所有 ATOM/HETATM 行 (去尾换行), 保留原序号
    header_lines : list[str]  REMARK/MODEL/ENDMDL 等非原子、非 CRYST1、非 TER/END
                              的前置记录行 (去尾换行); CONECT 行也在其中 (写出时丢弃)
    cryst1_line  : str|None   原始 CRYST1 行 (去尾换行), 无则为 None
    """
    atoms = []
    header_lines = []
    cryst1_line = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if _is_atom(line):
                atoms.append(line)
            elif line.startswith("CRYST1"):
                cryst1_line = line
            elif line.startswith("TER") or line.startswith("END"):
                continue
            else:
                # REMARK / MODEL / ENDMDL / CONECT / HEADER 等
                header_lines.append(line)
    return atoms, header_lines, cryst1_line


def _split_cif_row(line):
    """把一行 mmCIF 数据拆成 token 列表 (处理引号值)。

    规则 (简化自 CIF 1.1 语法, _atom_site 行足够): 单/双引号包裹的值可含
    空格, 闭合引号必须后跟空白或行尾; 行内 '#' (token 起始处) 之后为注释。
    """
    tokens = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c in " \t":
            i += 1
            continue
        if c == "#":
            break
        if c in "'\"":
            quote = c
            i += 1
            start = i
            while i < n and not (line[i] == quote
                                 and (i + 1 >= n or line[i + 1] in " \t")):
                i += 1
            tokens.append(line[start:i])
            i += 1  # 跳过闭合引号
        else:
            start = i
            while i < n and line[i] not in " \t":
                i += 1
            tokens.append(line[start:i])
    return tokens


def _read_cif_atom_site(path):
    """解析 mmCIF 的 _atom_site loop, 返回行字典列表; 无 _atom_site 时返回 None。

    只处理 loop_ 形式的 _atom_site (PDB 分发的 mmCIF 均为此格式)。行按出现
    顺序保留, 值已去引号; 字典键为标签去掉 '_atom_site.' 前缀的小写形式
    (如 'auth_asym_id' / 'label_comp_id')。不依赖任何第三方库。
    """
    with open(path) as f:
        lines = f.readlines()
    n = len(lines)
    i = 0
    while i < n:
        if not lines[i].startswith("loop_"):
            i += 1
            continue
        # 收集 loop 头部的标签行
        j = i + 1
        tags = []
        while j < n:
            s = lines[j].strip()
            if s.startswith("_"):
                tags.append(s.split()[0])
                j += 1
            else:
                break
        if tags and tags[0].lower().startswith("_atom_site."):
            names = [t.split(".", 1)[1].lower() for t in tags]
            rows = []
            while j < n:
                s = lines[j].strip()
                if not s:
                    j += 1
                    continue
                # loop 结束: 注释 / 新 loop / 新标签 / 数据块 / 多行文本字段
                if (s.startswith("#") or s.startswith("loop_")
                        or s.startswith("_") or s.startswith("data_")
                        or s.startswith("save_") or s.startswith("stop_")
                        or s.startswith(";")):
                    break
                toks = _split_cif_row(s)
                if len(toks) != len(names):
                    raise ValueError(
                        f"{path}: _atom_site 第 {j + 1} 行字段数 {len(toks)} "
                        f"与标签数 {len(names)} 不符: {s[:60]}"
                    )
                rows.append(dict(zip(names, toks)))
                j += 1
            return rows
        i = j
    return None


def _select_residue_lines(atoms, chain=None, resseq=None, resname=None):
    """从 atoms 中筛选满足条件的原子行 (AND 组合), 返回新 list。

    chain/resseq/resname 任一为 None 表示该条件通配 (不限)。
    三个参数均支持逗号分隔多值 (如 resname='Z4D,OH'), 任一值命中即算该条件通过。
    比较时去空格 (容忍 ' OH' vs 'OH')。
    """
    def _to_set(val):
        if val is None:
            return None
        return {str(v).strip() for v in str(val).split(",") if str(v).strip()}

    chain_set = _to_set(chain)
    resseq_set = _to_set(resseq)
    resname_set = _to_set(resname)

    out = []
    for ln in atoms:
        if chain_set is not None and ln[21] not in chain_set:
            continue
        if resseq_set is not None and ln[22:26].strip() not in resseq_set:
            continue
        if resname_set is not None and ln[17:20].strip() not in resname_set:
            continue
        out.append(ln)
    return out


def _make_ter(last_atom_line):
    """根据某条链最后一个原子行生成 TER 记录。

    TER 序号取该原子序号 (PDB 惯例: TER 终止前一个原子), 残基信息原样复制。
    """
    serial = int(last_atom_line[6:11])
    resname = last_atom_line[17:20]
    chain = last_atom_line[21]
    resseq = last_atom_line[22:26]
    return "TER   " + ("%5d" % serial) + "      " + resname + " " + chain + resseq


def _group_by_chain_resseq(atoms):
    """按 (chain, resSeq) 分组, 返回有序 dict {key: [lines]}, 保留出现顺序。"""
    groups = {}
    order = []
    for ln in atoms:
        key = (ln[21], ln[22:26])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(ln)
    return groups, order


def _renumber_serials(atoms, start=1):
    """对 atoms 行的 serial 列 (7-11) 重新连续编号, 返回新 list。"""
    out = []
    for i, ln in enumerate(atoms, start=start):
        out.append(ln[:6] + ("%5d" % i) + ln[11:])
    return out


def replace_residues(args):
    """把 source PDB 中指定残基整块替换到 target PDB 的同位置。

    匹配: 按 (chain, resseq) 定位 target 中的残基, 将其所有原子行删除,
    在该位置插入 source 中同 (chain, resseq) 残基的全部原子行。若 target
    中无匹配残基则报错 (避免静默丢失)。source 原子行原样搬运 (含 serial/
    HETATM 等标记), 不做重编号 -- 如需连续序号用 --renumber。
    CONECT 记录默认丢弃 (不重写映射), 需保留请改用专业工具。
    """
    target_atoms, target_header, target_cryst1 = _read_pdb(args.target)
    source_atoms, _, _ = _read_pdb(args.source)

    # 解析 --chain/--resseq 可为逗号分隔多值
    chains = [c.strip() for c in args.chain.split(",")] if args.chain else [None]
    resseqs = [r.strip() for r in args.resseq.split(",")] if args.resseq else [None]

    # 收集 source 侧待替换残基的原子行 (按匹配条件)
    src_groups, src_order = _group_by_chain_resseq(source_atoms)
    replacements = {}  # (chain, resseq) -> [source lines]
    for ch in chains:
        for rs in resseqs:
            # 对每个 (ch, rs) 在 source 中找匹配残基
            matched = [ln for ln in source_atoms
                       if (ch is None or ln[21] == ch)
                       and (rs is None or ln[22:26].strip() == rs)]
            if not matched:
                raise ValueError(
                    f"source 中未找到匹配残基 (chain={ch}, resseq={rs})"
                )
            key = (matched[0][21], matched[0][22:26])
            replacements[key] = matched

    # 在 target 中逐残基替换: 按残基分组遍历, 命中则替换, 否则保留原行
    tgt_groups, tgt_order = _group_by_chain_resseq(target_atoms)
    new_atoms = []
    replaced = set()
    for key in tgt_order:
        if key in replacements:
            new_atoms.extend(replacements[key])
            replaced.add(key)
        else:
            new_atoms.extend(tgt_groups[key])

    if len(replaced) != len(replacements):
        missing = set(replacements) - replaced
        raise ValueError(
            f"target 中未找到待替换残基: {missing} "
            f"(请检查 target 是否含对应 chain/resseq)"
        )

    if args.renumber:
        new_atoms = _renumber_serials(new_atoms)

    # 重建链边界 TER (按 chainID 切分)
    ter_lines = _build_ter_for_atoms(new_atoms)
    _write_text_pdb(args.output, target_header, new_atoms, ter_lines,
                    box=_parse_box(args.box) if args.box else target_cryst1)
    print(f"已替换 {len(replaced)} 个残基, "
          f"输出 {len(new_atoms)} 个原子 -> {args.output}", file=sys.stderr)


def append_residues(args):
    """把 source PDB 中指定残基追加到 target PDB 末端, 保留源 chain id。

    追加顺序与 source 中出现顺序一致。自动按 chainID 在追加段末尾补 TER。
    source 原子行原样搬运; 默认不重编号 (序号可能不连续), --renumber 可重排。
    CONECT 默认丢弃。
    """
    target_atoms, target_header, target_cryst1 = _read_pdb(args.target)
    source_atoms, _, _ = _read_pdb(args.source)

    src_lines = _select_residue_lines(
        source_atoms,
        chain=args.chain,
        resseq=args.resseq,
        resname=args.resname,
    )
    if not src_lines:
        raise ValueError(
            f"source 中未找到匹配残基 "
            f"(chain={args.chain}, resseq={args.resseq}, resname={args.resname})"
        )

    # 检查 chain id 冲突: 若追加的 chain 已存在于 target, 警告 (仍执行)
    tgt_chains = set(ln[21] for ln in target_atoms)
    src_chains = set(ln[21] for ln in src_lines)
    conflict = src_chains & tgt_chains
    if conflict and not args.allow_chain_conflict:
        raise ValueError(
            f"source 追加链 {conflict} 已存在于 target 中, "
            f"可能造成重复 chain id。用 --allow-chain-conflict 强制继续, "
            f"或先在 source 中改 chain id。"
        )
    elif conflict:
        print(f"警告: 追加链 {conflict} 与 target 已有链重复, 已强制继续",
              file=sys.stderr)

    # 合并: target 原子 + 追加原子
    all_atoms = list(target_atoms) + list(src_lines)
    if args.renumber:
        all_atoms = _renumber_serials(all_atoms)

    # 重建所有链的 TER (target 侧原有的 + 追加侧)
    ter_lines = _build_ter_for_atoms(all_atoms)
    _write_text_pdb(args.output, target_header, all_atoms, ter_lines,
                    box=_parse_box(args.box) if args.box else target_cryst1)
    print(f"已追加 {len(src_lines)} 个原子 ({src_chains}) 到末端, "
          f"输出共 {len(all_atoms)} 个原子 -> {args.output}", file=sys.stderr)


def _parse_box(box_str):
    """把 --box 字符串解析为 CRYST1 行。

    支持: 'A' / 'A,B,C' / 'A,B,C,alpha,beta,gamma' (长度 Å, 角度 度)。
    返回格式化的 CRYST1 行字符串。
    """
    parts = [p.strip() for p in box_str.split(",")]
    if len(parts) == 1:
        a = b = c = float(parts[0])
        alpha = beta = gamma = 90.0
    elif len(parts) == 3:
        a, b, c = (float(p) for p in parts)
        alpha = beta = gamma = 90.0
    elif len(parts) == 6:
        a, b, c, alpha, beta, gamma = (float(p) for p in parts)
    else:
        raise ValueError("--box 需为 1/3/6 个值 (A / A,B,C / A,B,C,alpha,beta,gamma)")
    return _format_cryst1(a, b, c, alpha, beta, gamma)


# 常见水/离子/溶剂残基名 (提取序列时静默跳过, 不视为蛋白残基)
_NON_PROTEIN_RESNAMES = {
    "HOH", "WAT", "DOD", "SOL", "TIP3",
    "NA", "CL", "K", "CA", "MG", "ZN", "MN", "FE", "CU", "CO", "NI",
    "SOD", "CLA", "POT", "CAL", "MAG", "ZIN",
}


def _records_from_pdb_atoms(atoms):
    """把 PDB 原子行转为通用残基记录 (chain, resseq, resname)。

    与 _extract_sequence_from_atoms 的取列规则一致:
    chainID=22 列, resSeq=23-26 列, iCode=27 列, resName=18-20 列。
    """
    return [(ln[21], ln[22:27], ln[17:20]) for ln in atoms]


def _records_from_cif_rows(rows):
    """把 _atom_site 行字典转为通用残基记录 (chain, resseq, resname)。

    chain 优先 auth_asym_id (与 PDB chainID 对应), 缺失时退回 label_asym_id;
    resseq 优先 auth_seq_number, 退回 label_seq_id; resname 用 label_comp_id
    (与 auth_comp_id 在 _atom_site 中通常一致)。iCode 取 pdbx_PDB_ins_code,
    '?'/'.' 表示无。含 altloc 时同一残基只保留首个出现的条目 (调用处按 key 去重)。
    """
    def col(row, *names):
        for nm in names:
            v = row.get(nm)
            if v is not None and v not in ("?", "."):
                return v
        return None

    records = []
    for row in rows:
        group = row.get("group_PDB", "ATOM")
        if group not in ("ATOM", "HETATM"):
            continue
        chain = col(row, "auth_asym_id", "label_asym_id") or " "
        resseq = col(row, "auth_seq_number", "label_seq_id") or ""
        resname = col(row, "label_comp_id", "auth_comp_id") or ""
        icode = col(row, "pdbx_PDB_ins_code") or " "
        records.append((chain, ("%4s%s" % (resseq, icode)), resname))
    return records


def _records_from_structure(path):
    """解析结构文件为通用残基记录, 按扩展名分发, 内容不符时自动回退。

    返回 (records, fmt), records 为 (chain, resseq_str, resname) 列表,
    fmt 为 'cif' 或 'pdb'。
    .cif/.mmcif 先尝试 mmCIF _atom_site 解析; 若无 _atom_site 但含
    ATOM/HETATM 行 (如被 OpenMM 等工具以 PDB 格式写出的 .cif), 回退按
    PDB 定长列解析。其余扩展名一律按 PDB 解析。
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        rows = _read_cif_atom_site(path)
        if rows is not None:
            return _records_from_cif_rows(rows), "cif"
        atoms, _, _ = _read_pdb(path)
        if atoms:
            return _records_from_pdb_atoms(atoms), "pdb"
        raise ValueError(
            f"{path}: 未找到 _atom_site loop 也没有 ATOM/HETATM 行 "
            f"(该文件可能不含坐标信息)"
        )
    atoms, _, _ = _read_pdb(path)
    return _records_from_pdb_atoms(atoms), "pdb"


def _sequences_from_records(records, chains=None):
    """从通用残基记录按链提取蛋白序列。

    按 (chain, resseq) 去重, 每残基取首个原子的 resName 映射为一字母代码,
    跳过水/离子与未识别残基 (后者计数入 unknown)。
    chains 非 None 时只保留指定链 (支持逗号分隔多值); 返回
    (ordered {chain: seq_str}, {chain: n_residues}, unknown 计数 dict)。
    """
    chain_set = None
    if chains:
        chain_set = {c.strip() for c in str(chains).split(",") if c.strip()}

    seqs = {}       # chain -> 一字母序列 (保持首次出现顺序)
    n_res = {}      # chain -> 残基数 (含跳过的水/离子, 供提示)
    unknown = {}    # resname -> count (未识别且非水/离子的残基)
    seen = set()
    for chain, resseq, resname in records:
        if chain_set is not None and chain not in chain_set:
            continue
        key = (chain, resseq)
        if key in seen:
            continue
        seen.add(key)
        resname = resname.strip().upper()
        if resname in _NON_PROTEIN_RESNAMES:
            continue
        one = _THREE_TO_ONE.get(resname)
        if one is None:
            unknown[resname] = unknown.get(resname, 0) + 1
            continue
        seqs.setdefault(chain, []).append(one)
        n_res[chain] = n_res.get(chain, 0) + 1
    return ({ch: "".join(v) for ch, v in seqs.items()}, n_res, unknown)


def extract_sequence(args):
    """提取一个或多个 PDB / mmCIF 的蛋白序列 (基于坐标 ATOM 记录), 写出 fasta。

    支持格式: .pdb 按 PDB 定长列文本解析; .cif/.mmcif 解析 _atom_site loop。
    每条链单独输出一条 fasta 记录; 用 --chains 限定只要哪些链 (如 'A' 或
    'A,B'), 不指定则输出所有含蛋白残基的链。fasta header 默认为文件名 (去
    扩展名); --id-format file+chain 时附链 id。序列按 --width 折行 (默认 60,
    0 表示不折行)。
    """
    records_out = []
    for path in args.input:
        atoms, fmt = _records_from_structure(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        seqs, _, unknown = _sequences_from_records(atoms, chains=args.chains)

        if unknown:
            print(f"警告: {stem} 中有未识别残基被跳过: {unknown}",
                  file=sys.stderr)
        if args.chains:
            missing = [c.strip() for c in args.chains.split(",")
                       if c.strip() and c.strip() not in seqs]
            if missing:
                print(f"警告: {stem} 中未找到链 {missing} (提取到: "
                      f"{','.join(seqs) or '无'})", file=sys.stderr)
        if not seqs:
            print(f"警告: {stem} 未提取到蛋白序列, 已跳过", file=sys.stderr)
            continue

        for ch, seq in seqs.items():
            if args.id_format == "file":
                header = stem
            else:  # file+chain
                header = f"{stem}|chain_{ch.strip() or '_'}"
            records_out.append((header, seq))
            print(f"{stem} 链 {ch}: {len(seq)} 残基", file=sys.stderr)

    if not records_out:
        raise ValueError("没有提取到任何序列")

    lines = []
    for header, seq in records_out:
        lines.append(f">{header}")
        if args.width > 0:
            lines.extend(seq[i:i + args.width]
                         for i in range(0, len(seq), args.width))
        else:
            lines.append(seq)

    text = "\n".join(lines) + "\n"
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)
        print(f"已写入 {len(records_out)} 条序列 ({fmt} 输入) -> {args.output}",
              file=sys.stderr)
    else:
        print(text, end="")


def renumber_residues(args):
    """把 PDB 每条 chain 的 resid 重编为从 start 开始连续递增 (文本级)。

    每条链独立编号: chain A -> start, start+1, ...; chain B -> start, start+1, ...
    同一残基 (chain, resSeq, iCode) 内所有原子保持同一新 resid。
    HETATM (配体/水/离子) 默认一并重编; --protein-only 时只重编 ATOM 记录,
    HETATM 保持原 resid 不变。
    可选输出旧->新 resid 映射表 (csv), 便于回查。
    """
    atoms, header_lines, cryst1 = _read_pdb(args.input)

    start = args.start
    mapping_rows = []  # (chain, old_resid, new_resid, resname)
    new_atoms = []
    # 每条链独立计数
    cur_chain = None
    cur_key = None          # (chain, resSeq, iCode) 当前残基
    cur_new = None          # 当前残基的新 resid
    next_id = {}            # chain -> 下一个可用新 resid

    for ln in atoms:
        is_het = ln.startswith("HETATM")
        chain = ln[21]
        key = (chain, ln[22:26], ln[26])
        if is_het and args.protein_only:
            new_atoms.append(ln)
            continue
        if key != cur_key:
            # 进入新残基
            if chain not in next_id:
                next_id[chain] = start
            cur_new = next_id[chain]
            next_id[chain] += 1
            cur_key = key
            mapping_rows.append((chain, ln[22:26].strip(), cur_new,
                                 ln[17:20].strip()))
        # 重写 resSeq 列 (23-26, 右对齐 4 字符), 清空 iCode (27 列)
        new_atoms.append(ln[:22] + ("%4d" % cur_new) + " " + ln[27:])
        cur_chain = chain

    ter_lines = _build_ter_for_atoms(new_atoms)
    _write_text_pdb(args.output, header_lines, new_atoms, ter_lines,
                    box=cryst1)

    n_res = len(mapping_rows)
    chains = sorted({m[0] for m in mapping_rows})
    print(f"已重编 {n_res} 个残基 (链 {chains}, 起始 {start}) -> {args.output}",
          file=sys.stderr)

    if args.map:
        import csv as _csv
        with open(args.map, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["chain", "old_resid", "new_resid", "resname"])
            w.writerows(mapping_rows)
        print(f"映射表 -> {args.map}", file=sys.stderr)


def set_box(args):
    """改写或插入 CRYST1 盒子记录。支持 --box A / A,B,C / A,B,C,alpha,beta,gamma。"""
    new_cryst1 = _parse_box(args.box)
    atoms, header_lines, _ = _read_pdb(args.input)
    _write_text_pdb(args.output, header_lines, atoms,
                    ter_lines=_build_ter_for_atoms(atoms),
                    box=new_cryst1)
    print(f"盒子已改写 -> {args.output}", file=sys.stderr)


# ---- 文本级操作的辅助函数 ----

def _format_cryst1(a, b, c, alpha=90.0, beta=90.0, gamma=90.0):
    """格式化 CRYST1 行 (长度单位 Å, 角度 度)。"""
    return ("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1 "
            % (a, b, c, alpha, beta, gamma))


def _build_ter_for_atoms(atoms):
    """按 chainID 分段, 在每段最后一个原子后生成一条 TER。

    返回 list[(insert_index, ter_line)], insert_index 为 TER 应插入到的
    atoms 索引之后 (即该链最后原子的位置)。
    """
    ter_lines = []  # (after_position_in_atoms, ter_str)
    if not atoms:
        return ter_lines
    prev_chain = atoms[0][21]
    for i in range(1, len(atoms)):
        if atoms[i][21] != prev_chain:
            # 前一条链在 i-1 结束
            ter_lines.append((i, _make_ter(atoms[i - 1])))
            prev_chain = atoms[i][21]
    # 最后一链
    ter_lines.append((len(atoms), _make_ter(atoms[-1])))
    return ter_lines


def _write_text_pdb(out_path, header_lines, atoms, ter_lines, box=None):
    """写出文本 PDB: header + CRYST1 + atoms(夹 TER) + END。CONECT 默认不写。"""
    with open(out_path, "w") as f:
        for ln in header_lines:
            # 过滤掉 CONECT (默认丢弃)
            if ln.startswith("CONECT"):
                continue
            f.write(ln + "\n")
        if box is not None:
            f.write(box + "\n")
        ter_map = {pos: ter for pos, ter in ter_lines}
        for i, ln in enumerate(atoms, start=1):
            f.write(ln + "\n")
            if i in ter_map:
                f.write(ter_map[i] + "\n")
        f.write("END\n")


# ---------------------------------------------------------------------------
# smd_info: 从已构建体系 (pdbx, 如 solv.pdbx / last.pdbx) 提取 SMD/约束所需信息
#
# restraint / smd 的原子索引是对**已构建体系**的 0-based index (与 OpenMM/mdtraj
# 一致), 不是原始 *_reordered.pdb 里的序号。本命令用 mdtraj 读 pdbx, 按
# (resSeq, atom_name) 在蛋白链内定位原子, 并计算所有蛋白 CA 的质心。
# 依赖 mdtraj (env 已装)。
# ---------------------------------------------------------------------------

def _require_mdtraj():
    try:
        import mdtraj as md
        import numpy as _np
    except ImportError as e:
        raise ImportError(f"smd_info 依赖 mdtraj 与 numpy: {e}")
    return md, _np


def smd_info(args):
    """从已构建体系 pdbx 提取原子 index 与 CA 质心 (0-based)。

    对每个 --target (格式 resSeq:ATOMNAME, 如 189:OG) 在蛋白链 (含 CA 的链)
    内找唯一匹配原子, 返回其 0-based index; 并输出全部蛋白 CA 的 index 列表
    及其质心 (nm)。结果以 JSON 打印 (或 -o 写文件), 供生成 smd/eq yaml。
    """
    md, np = _require_mdtraj()
    traj = md.load(args.input)
    top = traj.topology

    # 蛋白链 = 含 CA 原子的链 (排除水/离子/配体)
    ca_idx = top.select("name CA")
    protein_chains = sorted({top.atom(i).residue.chain.index for i in ca_idx})

    def find_atom(resseq, atom_name):
        hits = []
        for r in top.residues:
            if r.chain.index not in protein_chains:
                continue
            if r.resSeq != resseq:
                continue
            for a in r.atoms:
                if a.name == atom_name:
                    hits.append((a.index, r.name, r.resSeq, r.chain.index))
        return hits

    out = {"input": args.input, "protein_chains": protein_chains}
    # CA indices + centroid
    ca_list = sorted(int(i) for i in ca_idx)
    out["n_ca"] = len(ca_list)
    if args.ca_indices:
        out["ca_indices"] = ca_list
    if ca_list:
        com = np.mean(traj.xyz[0][ca_list], axis=0)
        out["ca_centroid_nm"] = [float(x) for x in com]

    # targets
    targets = {}
    for t in args.target or []:
        resseq_s, atom_name = t.split(":")
        resseq = int(resseq_s)
        hits = find_atom(resseq, atom_name)
        if len(hits) == 0:
            targets[t] = {"error": "not found"}
        elif len(hits) > 1:
            targets[t] = {"error": f"ambiguous ({len(hits)} hits)", "hits": hits}
        else:
            idx, rname, rseq, ch = hits[0]
            targets[t] = {"index": int(idx), "resname": rname,
                          "resSeq": int(rseq), "chain": int(ch)}
    out["targets"] = targets

    import json
    text = json.dumps(out, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
        print(f"已写入 -> {args.output}", file=sys.stderr)
    else:
        print(text)


def main():
    parser = argparse.ArgumentParser(
        description="蛋白结构处理工具集 (基于 MDAnalysis)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # align 子命令
    p_align = subparsers.add_parser(
        "align", help="将 mobile 结构叠合到 reference 结构"
    )
    p_align.add_argument("-m", "--mobile", required=True,
                         help="待对齐的输入结构 (.pdb)")
    p_align.add_argument("-r", "--reference", required=True,
                         help="参考结构 (.pdb)")
    p_align.add_argument("-o", "--output", required=True,
                         help="对齐后的输出文件 (.pdb)")
    p_align.add_argument("-s", "--select", default=DEFAULT_SELECT,
                         help=f"用于对齐的原子选择 (默认: '{DEFAULT_SELECT}')")
    p_align.add_argument("--select-mobile", default=None,
                         help="mobile 侧独立选择 (需与 --select-ref 同时使用)")
    p_align.add_argument("--select-ref", default=None,
                         help="reference 侧独立选择 (需与 --select-mobile 同时使用)")
    p_align.add_argument("--weights", default="none", choices=["mass", "none"],
                         help="叠合权重 (mass/none, 默认 none)")
    p_align.add_argument("--match-atoms", action=argparse.BooleanOptionalAction,
                         default=True,
                         help="是否按原子名匹配两侧原子 (默认开启, 用 --no-match-atoms 关闭)")
    p_align.add_argument("--all-frames", action="store_true", default=False,
                         help="对多帧 (多 MODEL) PDB 的所有帧进行对齐")
    p_align.set_defaults(func=align_structures)

    # align-seq 子命令 (基于序列比对的对齐, 处理序列不一致的链)
    p_aseq = subparsers.add_parser(
        "align-seq",
        help="基于序列比对的对齐 (处理序列不一致的链: 缺失/插入/HIS 变体)"
    )
    p_aseq.add_argument("-m", "--mobile", required=True,
                        help="待对齐的输入结构 (.pdb)")
    p_aseq.add_argument("-r", "--reference", required=True,
                        help="参考结构 (.pdb)")
    p_aseq.add_argument("-o", "--output", required=True,
                        help="对齐后的 mobile 全结构输出文件 (.pdb)")
    p_aseq.add_argument("--chain", default="A",
                        help="用于序列匹配的链 id (默认 A)")
    p_aseq.add_argument("--weights", default="none", choices=["mass", "none"],
                        help="叠合权重 (mass/none, 默认 none)")
    p_aseq.add_argument("--gap-open", type=float, default=-10.0,
                        help="序列比对 gap open penalty (默认 -10)")
    p_aseq.add_argument("--gap-extend", type=float, default=-0.5,
                        help="序列比对 gap extend penalty (默认 -0.5)")
    p_aseq.set_defaults(func=align_by_sequence)

    # rmsd 子命令
    p_rmsd = subparsers.add_parser("rmsd", help="计算两结构间 RMSD")
    p_rmsd.add_argument("-m", "--mobile", required=True, help="结构 1 (.pdb)")
    p_rmsd.add_argument("-r", "--reference", required=True, help="结构 2 (.pdb)")
    p_rmsd.add_argument("-s", "--select", default=DEFAULT_SELECT,
                        help=f"计算 RMSD 的原子选择 (默认: '{DEFAULT_SELECT}')")
    p_rmsd.add_argument("--weights", default="none", choices=["mass", "none"],
                        help="权重 (mass/none, 默认 none)")
    p_rmsd.set_defaults(func=compute_rmsd)

    # select 子命令
    p_sel = subparsers.add_parser(
        "select", help="按链/残基名/残基号/原子名等条件筛选原子, 返回 index"
    )
    p_sel.add_argument("-i", "--input", required=True, help="输入结构文件 (.pdb)")
    # 基础属性条件 (AND 组合)
    p_sel.add_argument("--chain", default=None,
                       help="链 id, 逗号分隔多值 (如 'A,B'); 对应 segid")
    p_sel.add_argument("--resname", default=None,
                       help="残基名, 逗号分隔多值 (如 'ALA,GLY')")
    p_sel.add_argument("--resid", default=None,
                       help="残基号, 逗号分隔, 支持区间 (如 '1,3,5-7')")
    p_sel.add_argument("--atom", default=None,
                       help="原子名, 逗号分隔多值 (如 'CA,CB')")
    p_sel.add_argument("--element", default=None,
                       help="元素符号, 逗号分隔多值 (如 'C,N')")
    # 便捷开关
    p_sel.add_argument("--protein", action="store_true", default=False,
                       help="仅蛋白原子")
    p_sel.add_argument("--nucleic", action="store_true", default=False,
                       help="仅核酸原子")
    heavy_grp = p_sel.add_mutually_exclusive_group()
    heavy_grp.add_argument("--heavy", action="store_true", default=False,
                           help="仅重原子 (排除 H*)")
    heavy_grp.add_argument("--no-h", action="store_true", default=False,
                           help="排除氢原子 (等同 --heavy)")
    # 空间邻域条件
    p_sel.add_argument("--around", nargs=2, metavar=("SELECT", "CUTOFF"), default=None,
                       help="距 SELECT 选择命中的原子 CUTOFF Å 以内的原子 "
                            "(SELECT 为 MDAnalysis 选择语句, 如 'resname LIG')")
    p_sel.add_argument("--point", nargs=4, metavar=("X", "Y", "Z", "CUTOFF"),
                       default=None, type=float,
                       help="距坐标 (X,Y,Z) CUTOFF Å 以内的原子")
    # 原始选择语句 (追加 AND)
    p_sel.add_argument("--raw", default=None,
                       help="追加任意 MDAnalysis 选择语句 (与其它条件 AND 组合)")
    # 输出
    p_sel.add_argument("-f", "--format", default="table",
                       choices=["table", "csv", "list"],
                       help="输出格式: table(默认)/csv/list(仅 index)")
    p_sel.add_argument("-o", "--output", default=None,
                       help="输出文件路径, 不指定则打印到 stdout")
    p_sel.set_defaults(func=select_atoms_query)

    # replace 子命令 (文本级残基替换)
    p_rep = subparsers.add_parser(
        "replace",
        help="把 source PDB 指定残基整块替换到 target PDB 同位置 (文本级)"
    )
    p_rep.add_argument("-t", "--target", required=True,
                       help="被修改的 target PDB")
    p_rep.add_argument("-s", "--source", required=True,
                       help="提供替换残基的 source PDB")
    p_rep.add_argument("--chain", required=True,
                       help="匹配链 id, 逗号分隔多值 (如 'A'); source/target 都按此匹配")
    p_rep.add_argument("--resseq", required=True,
                       help="匹配残基号, 逗号分隔多值 (如 '152'); source/target 都按此匹配")
    p_rep.add_argument("--renumber", action="store_true", default=False,
                       help="重排序号 1..N (默认保留原 serial)")
    p_rep.add_argument("--box", default=None,
                       help="同时改写盒子, 格式同 setbox --box")
    p_rep.add_argument("-o", "--output", required=True,
                       help="输出 PDB 路径")
    p_rep.set_defaults(func=replace_residues)

    # append 子命令 (文本级残基追加)
    p_app = subparsers.add_parser(
        "append",
        help="把 source PDB 指定残基追加到 target PDB 末端 (文本级)"
    )
    p_app.add_argument("-t", "--target", required=True,
                       help="被追加的 target PDB")
    p_app.add_argument("-s", "--source", required=True,
                       help="提供追加残基的 source PDB")
    p_app.add_argument("--chain", default=None,
                       help="筛选 source 链 id (如 'B'); 不指定则不限链")
    p_app.add_argument("--resseq", default=None,
                       help="筛选 source 残基号 (如 '1,2'); 不指定则不限")
    p_app.add_argument("--resname", default=None,
                       help="筛选 source 残基名 (如 'Z4D,OH'); 不指定则不限")
    p_app.add_argument("--allow-chain-conflict", action="store_true",
                       default=False,
                       help="允许追加链 id 与 target 已有链重复 (默认报错)")
    p_app.add_argument("--renumber", action="store_true", default=False,
                       help="重排序号 1..N (默认保留原 serial)")
    p_app.add_argument("--box", default=None,
                       help="同时改写盒子, 格式同 setbox --box")
    p_app.add_argument("-o", "--output", required=True,
                       help="输出 PDB 路径")
    p_app.set_defaults(func=append_residues)

    # setbox 子命令 (改写 CRYST1)
    p_box = subparsers.add_parser(
        "setbox", help="改写/插入 CRYST1 盒子记录 (文本级)"
    )
    p_box.add_argument("-i", "--input", required=True, help="输入 PDB")
    p_box.add_argument("--box", required=True,
                       help="盒子参数: A / A,B,C / A,B,C,alpha,beta,gamma (Å/度)")
    p_box.add_argument("-o", "--output", required=True, help="输出 PDB 路径")
    p_box.set_defaults(func=set_box)

    # seq 子命令 (从坐标记录提取蛋白序列 -> fasta)
    p_seq = subparsers.add_parser(
        "seq",
        help="从 PDB / mmCIF 坐标 (ATOM) 记录提取蛋白序列, 按链写出 fasta (文本级)"
    )
    p_seq.add_argument("-i", "--input", required=True, nargs="+",
                       help="输入结构文件 (.pdb / .cif / .mmcif), 可多个 (如 *.cif)")
    p_seq.add_argument("--chains", default=None,
                       help="只提取指定链 id (如 'A' 或 'A,B'); 不指定则输出所有链,"
                            " 每条链一条 fasta 记录")
    p_seq.add_argument("--id-format", default="file",
                       choices=["file", "file+chain"],
                       help="fasta header 格式: 仅文件名 / 文件名+链 (默认 file); "
                            "多链输入建议 file+chain")
    p_seq.add_argument("--width", type=int, default=60,
                       help="fasta 每行残基数 (默认 60; 0 表示不折行)")
    p_seq.add_argument("-o", "--output", default=None,
                       help="输出 fasta 路径, 不指定则打印到 stdout")
    p_seq.set_defaults(func=extract_sequence)

    # renum 子命令 (每条 chain 从 1 开始连续重编 resid)
    p_renum = subparsers.add_parser(
        "renum",
        help="每条 chain 从 start 开始连续重编 resid (文本级)"
    )
    p_renum.add_argument("-i", "--input", required=True, help="输入 PDB")
    p_renum.add_argument("-o", "--output", required=True, help="输出 PDB")
    p_renum.add_argument("--start", type=int, default=1,
                         help="每条链起始 resid (默认 1)")
    p_renum.add_argument("--protein-only", action="store_true", default=False,
                         help="只重编 ATOM 记录, HETATM(配体/水) 保持原 resid")
    p_renum.add_argument("--map", default=None,
                         help="可选, 输出旧->新 resid 映射 csv")
    p_renum.set_defaults(func=renumber_residues)

    # smd_info 子命令 (从已构建体系 pdbx 提取 atom index + CA 质心)
    p_smd = subparsers.add_parser(
        "smd_info",
        help="从已构建体系 pdbx 提取 restraint/SMD 所需 atom index (0-based) 与 CA 质心"
    )
    p_smd.add_argument("-i", "--input", required=True,
                       help="已构建体系坐标 (solv.pdbx / last.pdbx)")
    p_smd.add_argument("--target", action="append", default=None,
                       help="目标原子, 格式 resSeq:ATOMNAME (如 189:OG), 可多次")
    p_smd.add_argument("--ca-indices", action="store_true", default=False,
                       help="输出全部蛋白 CA 的 index 列表")
    p_smd.add_argument("-o", "--output", default=None,
                       help="输出 JSON 路径, 不指定则打印到 stdout")
    p_smd.set_defaults(func=smd_info)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
