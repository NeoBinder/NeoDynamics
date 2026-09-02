import argparse
import os

import numpy as np


def _read_indices_file(path, topo):
    """从文件读取原子索引, 每行一个整数, 1-based计数, 支持#注释"""
    indices = []
    with open(path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            for token in line.split(","):
                token = token.strip()
                if token:
                    indices.append(int(token) - 1)
    indices = np.unique(np.asarray(indices, dtype=int))
    if indices.size == 0:
        raise ValueError(f"索引文件为空: {path}")
    _check_indices(indices, topo.n_atoms)
    return np.sort(indices)


def _check_indices(indices, n_atoms):
    if indices.min() < 0 or indices.max() >= n_atoms:
        raise ValueError(
            f"原子索引超出范围: 体系共{n_atoms}个原子(1-{n_atoms}), "
            f"给定索引范围 {indices.min() + 1}-{indices.max() + 1}"
        )


def resolve_selection(selection, topo):
    """解析原子选择, 支持MDTraj选择语句或索引文件(1-based)"""
    # 若选择串是存在的文件路径, 直接按索引文件解析
    if os.path.isfile(selection):
        return _read_indices_file(selection, topo)
    try:
        indices = np.asarray(topo.select(selection), dtype=int)
    except Exception as e:
        raise ValueError(
            f"无法解析选择语句 '{selection}': {e}\n"
            "支持MDTraj选择语句(如 'protein', 'resname LIG and name CA') "
            "或索引文件路径(每行一个1-based原子序号)"
        )
    if indices.size == 0:
        raise ValueError(f"选择语句 '{selection}' 未匹配到任何原子")
    return indices


def _apply_stride(coords, stride):
    if stride <= 1:
        return coords
    return coords[::stride]


def _load_traj_subset(args, atom_indices, topo):
    """分chunk加载轨迹, 仅保留所需原子的坐标(单位nm), 降低内存占用"""
    import mdtraj as md

    coords_ls = []
    for chunk in md.iterload(
        args.trajectory, top=args.topology,
        chunk=args.chunk, atom_indices=atom_indices,
    ):
        coords_ls.append(chunk.xyz)
    if not coords_ls:
        raise ValueError(f"轨迹文件未读取到任何帧: {args.trajectory}")
    return np.concatenate(coords_ls)


def _make_vdw_radii(args, topo):
    """生成MDTraj默认的vdw半径表, 并应用用户修改"""
    from mdtraj.geometry.sasa import _ATOMIC_RADII

    radii = []
    for atom in topo.atoms:
        element = atom.element.symbol if atom.element is not None else None
        if element is None or element not in _ATOMIC_RADII:
            # MDTraj默认对未知元素按碳处理
            radii.append(_ATOMIC_RADII["C"])
        else:
            radii.append(_ATOMIC_RADII[element])
    radii = np.asarray(radii, dtype=np.float32)
    if args.change_radii:
        # 格式: "Cl:0.175,Na:0.2"  单位nm
        from mdtraj.core.element import get_by_symbol

        for item in args.change_radii.split(","):
            symbol, value = item.split(":")
            symbol, value = symbol.strip(), float(value)
            elem = get_by_symbol(symbol)
            mask = np.array(
                [
                    (atom.element is not None and atom.element == elem)
                    for atom in topo.atoms
                ]
            )
            radii[mask] = value
    return radii


def _sasa_worker(worker_args):
    """多进程worker: 计算一段帧的SASA(原子模式)"""
    coords_chunk, radii_with_probe, n_sphere_points = worker_args
    import mdtraj as md
    from mdtraj.geometry import _geometry

    n_frames, n_atoms = coords_chunk.shape[0], coords_chunk.shape[1]
    out = np.zeros((n_frames, n_atoms), dtype=np.float32)
    _geometry._sasa(
        coords_chunk,
        radii_with_probe,
        int(n_sphere_points),
        np.arange(n_atoms, dtype=np.int32),
        np.ones(n_atoms, dtype=np.int32),
        out,
    )
    return out


def _parallel_sasa(coords, radii_with_probe, n_sphere_points, n_jobs):
    """按帧分段, 多进程并行计算SASA(原子模式)"""
    import multiprocessing as mp

    n_frames = coords.shape[0]
    n_jobs = max(1, min(n_jobs, n_frames))
    # 按帧均匀分段
    splits = np.array_split(np.arange(n_frames), n_jobs)
    tasks = [
        (coords[idx].copy(), radii_with_probe, n_sphere_points)
        for idx in splits
        if idx.size > 0
    ]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.map(_sasa_worker, tasks)
    return np.concatenate(results, axis=0)


def sasa(args):
    """计算轨迹中所选原子的溶剂可及表面积(Shrake-Rupley算法)"""
    import mdtraj as md
    from mdtraj.geometry import _geometry

    # 1. 加载拓扑并解析原子选择
    topo = _load_topology(args)
    print(f"拓扑加载完成: {topo.n_atoms} atoms, {topo.n_residues} residues")

    atom_indices = resolve_selection(args.selection, topo)
    print(f"选择 '{args.selection}' -> {atom_indices.size} 个原子")

    # --ignore_hydrogen: 从选区中移除氢原子(可显著加速大蛋白体系)
    if args.ignore_hydrogen:
        h_mask = np.array(
            [atom.element.symbol == "H" for atom in topo.atoms]
        )
        removed = int(h_mask[atom_indices].sum())
        atom_indices = atom_indices[~h_mask[atom_indices]]
        print(f"--ignore_hydrogen: 移除{removed}个氢原子, 剩余{atom_indices.size}个")

    atom_radii = _make_vdw_radii(args, topo)

    # 2. 分chunk加载轨迹(仅保留选中原子, 大蛋白体系可显著降低内存)
    coords = _load_traj_subset(args, atom_indices, topo)
    coords = _apply_stride(coords, args.stride)
    n_frames = coords.shape[0]
    print(f"轨迹加载完成: {n_frames} 帧 x {coords.shape[1]} 个选中原子")

    # 3. 组装子轨迹(仅含选中原子, 半径表与之一一对应)
    sub_radii = atom_radii[atom_indices]
    sub_top = topo.subset(atom_indices)
    traj = md.Trajectory(coords, sub_top)

    # 4. 计算SASA
    # Shrake-Rupley算法: 在原子球面上均匀撒点, 统计未被其他原子覆盖的点数
    # 选中原子的SASA = 其自身球面面积 - 与体系中其他所有原子的接触遮挡面积
    # 若需要"蛋白自身"SASA, 请提供不含水/离子/小分子的蛋白体系
    # 若需要"蛋白-底物接触面"相关SASA, 请提供蛋白+底物复合物体系并分别选择
    n_sphere_points = args.n_sphere_points
    print(
        f"计算SASA: probe_radius={args.probe_radius} nm, "
        f"n_sphere_points={n_sphere_points}, mode={args.mode}"
    )
    if args.mode == "atom":
        radii = (sub_radii + args.probe_radius).astype(np.float32)
        if args.n_jobs > 1:
            print(f"多进程并行: {args.n_jobs} 个进程按帧分段计算")
            result = _parallel_sasa(
                traj.xyz, radii, n_sphere_points, args.n_jobs
            )
        else:
            atom_mapping = np.arange(traj.n_atoms, dtype=np.int32)
            atom_mask = np.ones(traj.n_atoms, dtype=np.int32)
            out = np.zeros((n_frames, traj.n_atoms), dtype=np.float32)
            _geometry._sasa(
                traj.xyz,
                radii,
                int(n_sphere_points),
                atom_mapping,
                atom_mask,
                out,
            )
            result = out
        labels = [f"atom_{i}" for i in atom_indices]
    else:
        # residue模式: 先算逐原子再按残基求和, 兼容多进程
        radii = (sub_radii + args.probe_radius).astype(np.float32)
        if args.n_jobs > 1:
            print(f"多进程并行: {args.n_jobs} 个进程按帧分段计算")
            atom_result = _parallel_sasa(
                traj.xyz, radii, n_sphere_points, args.n_jobs
            )
        else:
            atom_result = md.shrake_rupley(
                traj,
                probe_radius=args.probe_radius,
                n_sphere_points=n_sphere_points,
                mode="atom",
            )
        # 按残基聚合
        res_atom_indices = [
            [atom.index for atom in res.atoms]
            for res in traj.topology.residues
        ]
        result = np.stack(
            [atom_result[:, idx].sum(axis=1) for idx in res_atom_indices],
            axis=1,
        ).astype(np.float32)
        labels = [str(res) for res in traj.topology.residues]

    # 5. 输出
    total = result.sum(axis=1)
    header = "# frame\ttotal_nm2"
    data = np.column_stack([np.arange(n_frames), total])
    if args.per_atom:
        header += "\t" + "\t".join(labels)
        data = np.column_stack([data, result])
    np.savetxt(args.output, data, fmt="%.6f", header=header, comments="")
    print(f"SASA结果已保存: {args.output}")
    print(f"总SASA: mean={total.mean():.4f} nm^2, "
          f"min={total.min():.4f}, max={total.max():.4f}")


def _load_topology(args):
    """加载拓扑文件并返回mdtraj Topology对象及占位Trajectory"""
    import mdtraj as md

    if args.topology.endswith((".pdb", ".ent", ".cif", ".pdbx")):
        if args.topology.endswith((".cif", ".pdbx")):
            t = md.load(args.topology)
        else:
            t = md.load_pdb(args.topology)
    else:
        raise ValueError(
            f"拓扑文件格式不支持: {args.topology}, 仅支持.pdb/.cif/.pdbx"
        )
    return t.topology


def _has_real_time(traj):
    """判断轨迹是否携带真实时间信息(而非退化的0,1,2,...帧序号)

    mdtraj对XTC等格式保留每帧ps时间(含起始偏移与真实间隔),
    对DCD则退化为0,1,2,...。此处检测: 时间数组是否与"从0起、
    步长1"的等差序列不同 -> 视为携带真实时间。
    """
    if traj.time is None or traj.n_frames == 0:
        return False
    t = np.asarray(traj.time, dtype=float)
    if t.size == 1:
        return False
    # 退化的DCD时间: 整数序列 0,1,2,...(步长1, 单位ps无意义)
    expected = np.arange(t.size, dtype=float)
    return not np.allclose(t, expected)


def _resolve_ns_range(traj, start_ns, end_ns, dt):
    """根据时间(ns)范围换算帧索引区间, 闭区间

    优先使用轨迹自带的真实时间轴(traj.time, 单位ps); 若轨迹未保留
    时间(如DCD退化为0,1,2,...), 则回退到由--dt做线性换算
    (假设第0帧=0ns, 步长=dt ns)。未指定start/end_ns时返回None表示不裁剪。
    """
    if start_ns is None and end_ns is None:
        return None, None, None

    n_frames = traj.n_frames
    use_real_time = _has_real_time(traj)
    if use_real_time:
        # traj.time单位为ps, 转ns后二分查找
        time_ns = np.asarray(traj.time, dtype=float) / 1000.0
        t0, t1 = float(time_ns[0]), float(time_ns[-1])
        start_frame = 0 if start_ns is None else int(np.searchsorted(time_ns, start_ns, side="left"))
        end_frame = n_frames - 1 if end_ns is None else int(np.searchsorted(time_ns, end_ns, side="right")) - 1
        time_src = f"轨迹时间轴({t0:.4f}-{t1:.4f} ns, {n_frames}帧)"
    else:
        # 回退: 线性换算, 需用户提供--dt
        if dt is None:
            raise ValueError(
                "轨迹未保留真实时间(常见于DCD格式), 且未提供--dt, "
                "无法将--start_ns/--end_ns换算为帧索引。"
                "请加--dt(相邻帧时间间隔, 单位ns), 或改用--start_frame/--end_frame"
            )
        start_frame = 0 if start_ns is None else int(round(start_ns / dt))
        end_frame = n_frames - 1 if end_ns is None else int(round(end_ns / dt))
        time_src = f"--dt={dt} ns线性换算(假设第0帧=0ns)"

    if start_frame < 0:
        start_frame = 0
    if end_frame >= n_frames:
        end_frame = n_frames - 1
    if start_frame > end_frame:
        raise ValueError(
            f"时间范围换算后无效: start_frame={start_frame} > end_frame={end_frame}"
            f"(n_frames={n_frames}, start_ns={start_ns}, end_ns={end_ns}, "
            f"时间来源: {time_src})"
        )
    return start_frame, end_frame, time_src


def _make_atom_label(atom):
    """生成原子标签: chainId_resSeq_name, 如A_123_CA; 无chain则用'_'"""
    chain_id = atom.residue.chain.chain_id if atom.residue.chain.chain_id else "_"
    return f"{chain_id}_{atom.residue.resSeq}_{atom.name}"


def rmsf(args):
    """计算轨迹中所选原子的RMSF, 结果按原子存为CSV(索引为chainId_resSeq_name)"""
    import mdtraj as md
    import pandas as pd

    # 1. 加载拓扑并解析原子选择
    topo = _load_topology(args)
    print(f"拓扑加载完成: {topo.n_atoms} atoms, {topo.n_residues} residues")

    atom_indices = resolve_selection(args.selection, topo)
    print(f"选择 '{args.selection}' -> {atom_indices.size} 个原子")

    # 2. 解析对齐选区(用于去除平动/转动)
    #    RMSF计算前轨迹必须已去除整体平动转动(即"align后"状态)。
    #    默认由程序内部superpose到参考帧完成对齐: 通用且不易出错;
    #    若用户提供的是已对齐轨迹, 可用 --no_align 跳过以节省时间。
    align_ref_frame = args.align_ref_frame
    if args.no_align:
        align_selection = None
        print("已禁用程序内部对齐(--no_align): 假定输入轨迹已align完成")
    else:
        align_selection = resolve_selection(args.align_selection, topo)
        print(
            f"对齐选区 '{args.align_selection}' -> {align_selection.size} 个原子, "
            f"参考帧={align_ref_frame}"
        )

    # 3. 加载轨迹: 对齐时需同时载入选区原子+对齐原子, 否则仅载入选区原子
    load_indices = atom_indices
    if align_selection is not None:
        load_indices = np.union1d(atom_indices, align_selection).astype(int)
    traj = md.load(
        args.trajectory, top=args.topology,
        atom_indices=load_indices,
    )
    n_frames = traj.n_frames
    # 诊断时间来源: 优先用轨迹自带时间, 退化时回退到--dt
    real_time = _has_real_time(traj)
    if real_time:
        t_ns = np.asarray(traj.time, dtype=float) / 1000.0
        print(
            f"轨迹加载完成: {n_frames} 帧 x {traj.n_atoms} 个原子 "
            f"(携带真实时间: {t_ns[0]:.4f}-{t_ns[-1]:.4f} ns)"
        )
    else:
        msg = (f"轨迹加载完成: {n_frames} 帧 x {traj.n_atoms} 个原子 "
               f"(未保留真实时间")
        if args.dt is not None:
            msg += f", 将用--dt={args.dt} ns线性换算"
        msg += ")"
        print(msg)

    # 4. 按ns或帧范围裁剪
    start_frame, end_frame, time_src = _resolve_ns_range(
        traj, args.start_ns, args.end_ns, args.dt,
    )
    if args.start_frame is not None or args.end_frame is not None:
        sf = 0 if args.start_frame is None else args.start_frame
        ef = n_frames - 1 if args.end_frame is None else args.end_frame
        if sf < 0:
            sf = 0
        if ef >= n_frames:
            ef = n_frames - 1
        if sf > ef:
            raise ValueError(
                f"帧范围无效: start_frame={sf} > end_frame={ef}"
            )
        start_frame, end_frame = sf, ef
        print(
            f"按帧范围裁剪: 帧{start_frame}-{end_frame}, "
            f"共{end_frame - start_frame + 1}帧用于计算"
        )
    elif start_frame is not None:
        print(
            f"按时间范围裁剪: {args.start_ns}-{args.end_ns} ns "
            f"-> 帧{start_frame}-{end_frame}, "
            f"共{end_frame - start_frame + 1}帧用于计算 "
            f"[来源: {time_src}]"
        )
    if start_frame is not None:
        traj = traj[start_frame:end_frame + 1]

    # 5. 对齐: superpose到参考帧(去除整体平动转动), 仅用对齐选区拟合
    if align_selection is not None:
        # atom_indices在load后的局部索引需重映射
        idx_map = {old: new for new, old in enumerate(load_indices)}
        align_local = np.array(
            sorted(idx_map[i] for i in align_selection), dtype=int,
        )
        ref = traj[align_ref_frame]
        traj.superpose(
            ref, frame=0, atom_indices=align_local,
        )

    # 6. 选取用于RMSF计算的原子(在当前load后的局部索引空间)
    idx_map = {old: new for new, old in enumerate(load_indices)}
    rmsf_local = np.array(
        sorted(idx_map[i] for i in atom_indices), dtype=int,
    )
    sub_traj = traj.atom_slice(rmsf_local)

    # 7. 计算RMSF: 相对平均位置的原子位置涨落(nm)
    #    reference=None 表示以各原子自身的时间平均位置为参考, 即标准RMSF定义。
    #    对齐(superpose)已在上一步完成, 此处直接算RMSF。
    rmsf_values = md.rmsf(sub_traj, None)

    # 8. 组装结果: index = chainId_resSeq_name
    #    注意: mdtraj在atom_indices/atom_slice子集化拓扑后会丢失chain_id
    #    (变为None), 故标签必须从原始完整拓扑topo提取。
    #    atom_indices来自resolve_selection, 已由np.sort排序, 与rmsf_local
    #    的顺序一致(sub_traj保留load_indices中的相对顺序)。
    labels = [_make_atom_label(topo.atom(i)) for i in atom_indices]
    df = pd.DataFrame(
        {"rmsf_nm": rmsf_values, "rmsf_angstrom": rmsf_values * 10.0},
        index=labels,
    )
    df.index.name = "chainId_resSeq_atomName"
    df.to_csv(args.output)
    print(f"RMSF结果已保存: {args.output}")
    print(
        f"RMSF(nm): mean={df['rmsf_nm'].mean():.4f}, "
        f"min={df['rmsf_nm'].min():.4f}, max={df['rmsf_nm'].max():.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="轨迹处理工具集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SASA计算说明:
  采用Shrake-Rupley算法(在原子vdw球面+探针半径上均匀撒点,
  统计未被其他原子遮挡的点数), 结果已自动扣除与体系中其他
  所有原子的接触面积。因此:
    - 需要蛋白自身SASA: 提供不含水/离子/小分子的纯蛋白体系
    - 需要蛋白与底物接触相关的SASA: 提供蛋白+底物复合物体系,
      分别对蛋白和底物运行本命令后做差
  大蛋白/长轨迹体系建议调小 --chunk 控制内存, 并用
  --ignore_hydrogen / --stride 加速计算; 多核机器可用
  --n_jobs N 按帧分段并行计算。

RMSF计算说明:
  RMSF衡量每个原子相对其平均位置的涨落, 反映位置柔性。
  计算前轨迹必须已去除整体平动和转动(即"对齐/align后")。
  本命令默认在程序内部用 superpose 对齐到指定参考帧:
    --align_selection 指定对齐所用原子(默认"name CA"),
    --align_ref_frame 指定参考帧序号(默认0, 即首帧),
    --no_align 跳过程序内部对齐(仅当输入轨迹已align时使用)。
  如需仅计算某一时间段, 用 --start_ns/--end_ns 指定:
    - XTC等携带真实时间的轨迹: 直接按时间轴定位帧, 无需--dt;
    - DCD等未保留时间的轨迹: 需额外提供--dt(相邻帧间隔ns)做线性换算;
    - 或直接用 --start_frame/--end_frame 按帧索引指定。
  运行时会打印时间来源与换算结果, 可据此核对帧-时间映射是否正确。
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_sasa = subparsers.add_parser(
        "sasa", help="计算轨迹中原子选择的溶剂可及表面积(SASA)"
    )
    parser_sasa.add_argument(
        "-p", "--topology", required=True,
        help="拓扑/结构文件(.pdb/.cif/.pdbx), 应与轨迹对应"
    )
    parser_sasa.add_argument(
        "-t", "--trajectory", required=True,
        help="轨迹文件(.dcd/.xtc)"
    )
    parser_sasa.add_argument(
        "-s", "--selection", required=True,
        help='原子选择: MDTraj选择语句(如"protein", "resname LIG")'
             "或索引文件路径(每行一个1-based原子序号, 支持#注释)"
    )
    parser_sasa.add_argument(
        "-o", "--output", default="sasa.dat",
        help="输出文件路径(默认sasa.dat), 列为frame/total及可选的逐原子/残基值"
    )
    parser_sasa.add_argument(
        "--probe_radius", type=float, default=0.14,
        help="探针半径(nm), 默认0.14(对应水分子)"
    )
    parser_sasa.add_argument(
        "--n_sphere_points", type=int, default=960,
        help="球面采样点数, 越大越精确但越慢(默认960)"
    )
    parser_sasa.add_argument(
        "--mode", choices=["atom", "residue"], default="atom",
        help="输出分辨率: atom逐原子 / residue逐残基(默认atom)"
    )
    parser_sasa.add_argument(
        "--per_atom", action="store_true", default=False,
        help="输出中附带逐原子(或逐残基)的SASA列"
    )
    parser_sasa.add_argument(
        "--stride", type=int, default=1,
        help="每隔多少帧取一帧计算(默认1, 即全部帧)"
    )
    parser_sasa.add_argument(
        "--chunk", type=int, default=100,
        help="每次读入内存的帧数(默认100), 大体系可调小以节省内存"
    )
    parser_sasa.add_argument(
        "--ignore_hydrogen", action="store_true", default=False,
        help="忽略氢原子(可显著加速大蛋白体系计算)"
    )
    parser_sasa.add_argument(
        "--n_jobs", type=int, default=1,
        help="并行进程数(默认1)。>1时按帧分段, 多进程并行计算后合并结果"
    )
    parser_sasa.add_argument(
        "--change_radii", default=None,
        help='修改元素vdw半径, 格式"符号:半径nm", 逗号分隔, 如"Cl:0.175,Na:0.2"'
    )
    parser_sasa.set_defaults(func=sasa)

    parser_rmsf = subparsers.add_parser(
        "rmsf", help="计算轨迹中原子选择的RMSF(均方根涨落), 输出CSV"
    )
    parser_rmsf.add_argument(
        "-p", "--topology", required=True,
        help="拓扑/结构文件(.pdb/.cif/.pdbx), 应与轨迹对应"
    )
    parser_rmsf.add_argument(
        "-t", "--trajectory", required=True,
        help="轨迹文件(.dcd/.xtc)"
    )
    parser_rmsf.add_argument(
        "-s", "--selection", required=True,
        help='计算RMSF的原子选择: MDTraj选择语句(如"protein", '
             '"name CA", "resname LIG")或索引文件路径(每行一个1-based原子序号)'
    )
    parser_rmsf.add_argument(
        "-o", "--output", default="rmsf.csv",
        help="输出CSV路径(默认rmsf.csv), 列为rmsf_nm/rmsf_angstrom, "
             "索引为chainId_resSeq_atomName"
    )
    parser_rmsf.add_argument(
        "--align_selection", default="name CA",
        help='对齐所用原子选择(默认"name CA"), 仅在未--no_align时生效'
    )
    parser_rmsf.add_argument(
        "--align_ref_frame", type=int, default=0,
        help="对齐参考帧序号(默认0, 即首帧), 仅在未--no_align时生效"
    )
    parser_rmsf.add_argument(
        "--no_align", action="store_true", default=False,
        help="跳过程序内部对齐; 仅当输入轨迹已align时使用, "
             "否则RMSF结果将包含整体平动/转动而无意义"
    )
    parser_rmsf.add_argument(
        "--start_ns", type=float, default=None,
        help="计算起始时间(ns)。轨迹携带真实时间时直接按时间轴定位;"
             "否则需配合--dt线性换算。不指定则从首帧开始"
    )
    parser_rmsf.add_argument(
        "--end_ns", type=float, default=None,
        help="计算结束时间(ns)。轨迹携带真实时间时直接按时间轴定位;"
             "否则需配合--dt线性换算。不指定则到末帧"
    )
    parser_rmsf.add_argument(
        "--dt", type=float, default=None,
        help="相邻两帧时间间隔(ns), 仅当轨迹未保留真实时间(如DCD)时"
             "用于--start_ns/--end_ns的线性换算"
    )
    parser_rmsf.add_argument(
        "--start_frame", type=int, default=None,
        help="计算起始帧(0-based), 优先于--start_ns; 不指定则从首帧开始"
    )
    parser_rmsf.add_argument(
        "--end_frame", type=int, default=None,
        help="计算结束帧(0-based, 含), 优先于--end_ns; 不指定则到末帧"
    )
    parser_rmsf.set_defaults(func=rmsf)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
