# OPES（method: opes）

> 需求与设计决策：[issue #11](https://github.com/NeoBinder/NeoDynamics/issues/11)
> （路径 B 自研；openmm-plumed 路线被否决）

## 原理简述

OPES（On-the-fly Probability Enhanced Sampling）在线用加权 KDE 估计已
探索的概率分布，据此构造准静态偏差——well-tempered metadynamics 的
现代升级：摆脱网格存储（每加 hill 更新整个网格、网格随 CV 维数指数
增长），参数只有 `pace` / `barrier` / 核宽三项，并有 explore（均匀
探索目标，耐次优 CV）与 flooding 等变体。本实现与 metadynamics
triple 完全同构：每 `pace` 步存入一个（压缩后的）核、刷新归一化
Z_n、经同一 seam 推送新 bias 表；完整数学见 Invernizzi–Parrinello
2020/2022 论文与 PLUMED `OPES_METAD` 文档。

## 使用

OPES 与 metadynamics 共用同一 facade：`method: opes` + 相同的 `colvars:`
段（网格即 bias 表定义域，各 CV 的 `biasWidth` 即初始核宽 σ(0)）+
`opes_set:` 段。`opes_set` 只收三个输入——`pace`（偏差更新间隔步数）、
`barrier`（预期自由能垒，kJ/mol）、可选 `mode: standard`（默认，
收敛导向的 well-tempered 目标）或 `mode: explore`（均匀探索目标）。
γ、ε 与核截断均由 `barrier` 推导；**没有** `biasFactor`/`height` 键。
最小可运行 plan（单个 distance CV）：

```yaml
method: opes
steps: 50000
temperature: 298
seed: 2026
integrator: {dt: 0.002, friction_coeff: 1.0}
input_files:
  complex: /work_dir/min/last.pdbx
  system: /work_dir/sys_prep/htb/system.xml
output:
  output_dir: /work_dir/opes
  trajectory_interval: 1000
  checkpoint_interval: 1000
colvars:
  dist:
    type: distance
    grp1_idx: "0"
    grp2_idx: "1"
    min_cv_nm: 0.2          # bias 表定义域
    max_cv_nm: 1.2
    biasWidth_nm: 0.05      # 初始核宽 σ(0)
    bins: 200
opes_set:
  pace: 500                 # 偏差更新间隔（步）
  barrier: 25.0             # 预期自由能垒，kJ/mol
  # mode: explore           # 或 standard（默认）
```

## 产物

- **`colvar.tsv`**：CV 轨迹（自然单位，角 CV 为度）。
- **`kernels.npz`**：核 ledger（压缩前存入）——resume 回放态，经同一
  deposit 数学回放，核与直跑 bit-identical（同 metadynamics 的
  hills 回放）。
- **`fes.tsv`**：run 结束的 FES（estimator 按 mode：standard
  `-(1/β)log P_n`，explore `-γ/β·log p^WT_n`）。
- **与 analysis 的衔接**：`neomd analysis` 读回做收敛差值、block
  averaging、TP-reweight、multi-walker merge。

## 参考文献

- Invernizzi & Parrinello, *J. Phys. Chem. Lett.* 11, 2731（2020）——
  [OPES 原始论文](https://doi.org/10.1021/acs.jpclett.0c00497)。
- Ray & Parrinello, *J. Chem. Theory Comput.*（2022）—— OPES-explore；
  explore 目标分布出处为 Invernizzi, Piaggi & Parrinello,
  *J. Chem. Theory Comput.* 18, 3988（2022）。
- [PLUMED 文档](https://www.plumed.org/doc)：`OPES_METAD` /
  `OPES_METAD_EXPLORE` / `OPES_FLOODING`（实现期仅作文档参照，
  未复制代码）。
- [issue #11](https://github.com/NeoBinder/NeoDynamics/issues/11) ——
  需求与路径 B 决策。
