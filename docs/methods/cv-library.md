# CV 库：rmsd / coordination / path_s / path_z（colvars.py）

> 需求：[issue #14](https://github.com/NeoBinder/NeoDynamics/issues/14)
> （残留部分——funnel 已存在，rmsd 已是 restraint 形态）

## 原理简述

在 5 种表达式 CV（`distance` / `distance_ref` / `min_distances` /
`dihedral` / `angle`）之外新增 4 个 kind-driven CV：`rmsd`（Kabsch
最优旋转 RMSD）、`coordination`（两原子团配位数，有理切换函数）、
`path_s` / `path_z`（路径进度/偏离，Branduardi–Gervasio–Parrinello
定义，共享同一 spec 块语法）。每个 CV 是 `colvars.py` 里的一个知识
三元组（schema + kernel 表达式 + numpy `evaluate` 观测量），
`CVIR.kind` 驱动编译分发；kernel 字符串与表示约定记在
`colvars.py` 模块 docstring。

## 使用

各 CV 的 YAML 拼写（`colvars:` 列表项）：

```yaml
method: metadynamics
colvars:
  - cv_type: rmsd            # Kabsch 最优旋转 RMSD（nm 网格）
    ref_pos_file: ref.pdbx   # 全体系参考坐标（每粒子一行，nm）
    restr_grp: [10, 11, 12, 15]
    min_cv_nm: 0.0
    max_cv_nm: 1.0
    biasWidth_nm: 0.02
    bins: 50

  - cv_type: coordination    # 两原子团间的配位数（无量纲网格）
    grp1_idx: [10, 11, 12]
    grp2_idx: [40, 41]
    r0: 0.35                 # 参考距离（nm）
    nn: 6                    # 切换函数分子指数（默认 s(r)=1/(1+(r/r0)^6)）
    mm: 12                   # 分母指数
    min_cv: 0.0
    max_cv: 8.0
    biasWidth: 0.5
    bins: 40

  - cv_type: path_s          # 路径进度（无量纲网格）
    ref_path_file: path.pdb  # 多模型参考帧（MODEL/ENDMDL 或 pdbx_PDB_model_num，>=2 帧）
    restr_grp: [10, 11, 12]
    lambda: 0.35             # 平滑长度（nm），权重 exp(-MSD/lambda^2)
    min_cv: 0.0
    max_cv: 1.0
    biasWidth: 0.05
    bins: 50
  # path_z 同 ref_path_file/restr_grp/lambda，但为 nm 网格（min_cv_nm/...）
```

grid 约定：`rmsd`、`path_z` 用 nm 后缀键（`min_cv_nm`/`biasWidth_nm`），
`coordination`、`path_s` 无量纲（`min_cv`/`biasWidth`）。
`neomd.colvars` 暴露 numpy `evaluate` 观测量（报告几何用），与内核
编译路径双轨并存、逐位对拍钉死（`tests/v2/test_colvars_w1b.py`）。

## 产物

CV 值进 `colvar.tsv`（自然单位），与其它 CV 无差别；bias/分析按所配
method 的产物语义。

## 参考文献

- Branduardi, Gervasio & Parrinello, *J. Chem. Phys.* 126, 054103
  （2007）—— [path CV（s, z）](https://doi.org/10.1063/1.2432340)。
- [PLUMED 文档](https://www.plumed.org/doc)：PATH / COORDINATION。
- Limongelli, Bonomi & Parrinello, *PNAS* 110, 6358（2013）——
  [funnel metadynamics](https://doi.org/10.1073/pnas.1308648110)
  （背景；`funnel` restraint triple 已独立存在）。
