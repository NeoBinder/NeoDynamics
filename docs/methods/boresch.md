# Boresch Restraint（取向锚，restraint: boresch）

> 需求：[issue #8](https://github.com/NeoBinder/NeoDynamics/issues/8)（先行切片）·
> 选型：[ADR-0003](../adr/0003-rbfe-technology-selection.md) ·
> 姊妹文档：[rbfe.md](rbfe.md)（RBFE λ 窗口引擎）

## 原理简述

自由能计算的标准取向锚：用 3+3 个锚原子（受体 a1/a2/a3、配体
b1/b2/b3）上的 6 个几何分量——1 距离 + 2 角 + 3 二面角——把配体的
平动与转动都 restrain 住，否则去耦合后配体会飘走、采样发散。
r 与两个角取谐和形式，三个二面角取周期安全形式 `(k/2)(1 - cos(phi - phi0))`
（裸二次型会跨 ±180° 卷绕发散）；近平衡下二面角项等效二次常数
k/2 kJ/mol/rad²——RBFE 引擎的标准态/解析校正工作必须计入。

## 使用

`boresch` 出现在 plan 的 `restraint:` 段（任何 method 下都可用；
RBFE 场景里与 `method: rbfe` 的窗口 plan 同文件共存，见 `rbfe.md`）：

```yaml
method: eq
steps: 50000
temperature: 298
seed: 2026
integrator: {dt: 0.002, friction_coeff: 1.0}
input_files:
  complex: /work_dir/min/last.pdbx
  system: /work_dir/sys_prep/htb/system.xml
output:
  output_dir: /work_dir/eq_boresch
restraint:
  boresch_lig:
    type: boresch
    # 3+3 anchor atoms: receptor a1/a2/a3, ligand b1/b2/b3
    rec_grp1: "41,42,43"        # a1 (index list or single index)
    rec_grp2: "48"              # a2
    rec_grp3: "55,56"           # a3 — the receptor anchor apex
    lig_grp1: "260"             # b1
    lig_grp2: "261"             # b2
    lig_grp3: "262"             # b3 — the ligand anchor apex
    r0_nm: 0.45                 # equilibrium distance a3–b3
    thetaA0_degree: 60.0        # equilibrium angle a1–a3–b3
    thetaB0_degree: 50.0        # equilibrium angle a3–b3–b1
    phiA0_degree: 30.0          # equilibrium dihedral a1–a2–a3–b3
    phiB0_degree: 60.0          # equilibrium dihedral a2–a3–b3–b1
    phiC0_degree: 45.0          # equilibrium dihedral a3–b3–b1–b2
    restr_k_r: 20.0             # kJ/mol per nm^2 (bare kJ/mol, project convention)
    restr_k_theta: 20.0         # kJ/mol per rad^2, BOTH angles share
    restr_k_phi: 20.0           # kJ/mol, ALL three dihedrals share
    is_periodic: true           # optional, default true
```

六个锚组与六个平衡值一一对应 Boresch 2003 的记号
（r = distance(a3,b3)；θA = angle(a1,a3,b3)、θB = angle(a3,b3,b1)；
φA/φB/φC 为 a1-a2-a3-b3 / a2-a3-b3-b1 / a3-b3-b1-b2）；三个力常数各
覆盖同类全部分量。spec 键由 `plan.py` 做 collect-all 校验
（key path + did-you-mean）。

## 产物

- **`restraint.tsv`**：六个几何观测量（`r` nm，θ/φ 以度报告）+
  bias-energy 列，随 resume 通用规则截断。

## 参考文献

- Boresch, S.; Karplus, M. et al. *Absolute Binding Free Energies: A
  Quantitative Approach for Their Calculation.* J. Phys. Chem. B 2003,
  107, 9535–9551 ——
  [DOI](https://doi.org/10.1021/jp0217839)（势能形式 eq. 3.4 适配；
  周期安全二面角为 GROMACS/YANK 拼写）。
- [ADR-0003：RBFE 引擎技术选型](../adr/0003-rbfe-technology-selection.md)
  （openmmtools 走 prepare 边界、不自研 softcore、不整体引入 OpenFE）。
- [rbfe.md](rbfe.md)（λ 窗口引擎——run_ladder、du.tsv、BAR/MBAR）。
