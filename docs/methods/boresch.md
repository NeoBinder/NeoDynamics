# Boresch Restraint（取向锚，restraint triple）

> 状态：issue #8（先行切片）· 实现状态：已实装
> （`src/neomd/restraints.py` 的 `boresch` triple，10 个内置 restraint 之一）·
> ADR：[ADR-0003](../adr/0003-rbfe-technology-selection.md)（RBFE 选型）·
> 姊妹文档：[rbfe.md](rbfe.md)（RBFE λ 窗口引擎）

## 背景与动机

issue #8（RBFE 模块）需要一个标准锚把配体相对受体固定住，否则去耦合
（decoupling）后配体会飘走、采样发散。Boresch orientation restraint 是
自由能计算的经典方案：用 3+3 个锚原子（受体 a1/a2/a3、配体 b1/b2/b3）
上的 6 个几何分量（1 距离 + 2 角 + 3 二面角）把配体的平动与转动都
 restrain 住。

issue #8 原文设想由 v1 的约束构造器扩展自动生成——当前约束没有
constructor，而是 registry 里的
knowledge triple（schema + 力表达式 + observables）。本切片就是
`boresch` triple 本身；无 v1 先例，物理直接取自一手文献——Boresch, Karplus et al., J. Phys.
Chem. B 2003, 107, 9535–9551（issue #8 将本切片列为
"Boresch restraint triple（可先行）"，即 RBFE 引擎的解锁
前置之一）。

**本切片只含 restraint triple，不含 RBFE 引擎**：λ 窗口、
alchemical 扰动、du 带、BAR/MBAR 见 `rbfe.md`（互相链接）。

## 与 issue 方案的差异

- **载体**：一个标准
  restraint knowledge triple，经 `registry.register("restraint",
  "boresch", …)` 注入；spec 键由 `plan.py` 对 registry schema 做
  collect-all 校验（缺必填 + 未知键附 did-you-mean）。
- **物理出处**：无 v1 先例，属新物理，不走"逐字移植"（settled decision
  #2 的常规路径），而是从 Boresch 2003 一手文献实现：r 与两个角取
  `(k/2)(x - x0)^2` 谐和形式（表达式内角度用弧度，同 `angle`
  类型的 deg 声明惯例）；三个二面角取周期安全形式
  `(k/2)(1 - cos(phi - phi0))`（GROMACS/YANK 拼写；裸二次型会跨
  ±180° 卷绕发散）。近平衡下二面角项等效二次常数 k/2 kJ/mol/rad²
  ——RBFE 引擎的标准态/解析校正工作必须计入。
- **打包**：与 `distances` 同款 multi-bond 打包——每种表达式 KIND 一支
  `CustomCentroidBondForce`（[distance, angle, torsion] 共 3 支力，
  共享 32 力组预算），per-bond 参数走 `BiasIR.bonds`/`BondIR`
  （theta0/phi0 内部用弧度，r0 用 nm，k 用 bare kJ/mol）。
- **RBFE 引擎的其余部分**（softcore、λ 编排、BAR/MBAR）按 ADR-0003
  的分层落点留在 `rbfe.md`；openmmtools 只进 `rbfe` pixi env。

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

六个锚组与六个平衡值一一对应 Boresch 2003 的记号；三个力常数各覆盖
同类全部分量（`restr_k_theta` 管 θA/θB，`restr_k_phi` 管 φA/φB/φC）。
运行时与其他 restraint 一样被报告到 `restraint.tsv`：六个几何观测量
（`r` nm，`thetaA/thetaB/phiA/phiB/phiC` 以度报告——弧度只在表达式
内部）加 bias-energy 列（`RestraintProbe` + `GroupEnergy`，双轨报告
discipline #5）。

## 架构与产物

- **6 分量 · 3+3 锚原子**：r = distance(a3,b3)；θA = angle(a1,a3,b3)、
  θB = angle(a3,b3,b1)；φA = dihedral(a1,a2,a3,b3)、φB =
  dihedral(a2,a3,b3,b1)、φC = dihedral(a3,b3,b1,b2)。
- **3 支力**（与 `distances` 同款打包）：每表达式 KIND 一支
  `CustomCentroidBondForce`——distance 1 bond、angle 2 bonds、
  torsion 3 bonds，per-bond 参数经 `BondIR` 下发；力组 id 来自唯一
  分配器 `port.pick_free_force_group`。
- **产物**：`restraint.tsv`（六个几何观测量 + `__energy`），随
  resume 通用规则截断。

## 参考文献与 ADR

- Boresch, S.; Karplus, M. et al. *Absolute Binding Free Energies: A
  Quantitative Approach for Their Calculation.* J. Phys. Chem. B 2003,
  107, 9535–9551（势能形式 eq. 3.4 适配；周期安全二面角为
  GROMACS/YANK 拼写）
- [ADR-0003：RBFE 引擎技术选型](../adr/0003-rbfe-technology-selection.md)
  （openmmtools 走 prepare 边界、
  不自研 softcore、不整体引入 OpenFE）
- [rbfe.md](rbfe.md)（λ 窗口引擎——run_ladder、du.tsv、
  BAR/MBAR）
