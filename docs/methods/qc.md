# 结构质量校验（neomd.qc）

> 需求：[issue #15](https://github.com/NeoBinder/NeoDynamics/issues/15)（含
> [#7](https://github.com/NeoBinder/NeoDynamics/issues/7) 回归）

## 原理简述

建系后、min 后自动执行结构质量检查（纯 numpy 几何，不经 kernel
port）：NaN/Inf 坐标、出盒、PBC 感知最小镜像 clash（排除 1-2/1-3/1-4
键连对）、键长对 `r0` 偏离、键角对 `theta0` 偏离；体系带
`input_files.ligands` 时同一套检查按配体范围再跑一遍。检查项
collect-all，报告全部 findings 后统一裁决——默认 `soft` 仅报告，
`strict` 抛 `StructureQualityError`。阈值用 issue #7 复现数据标定
（坏 minimize 留下键长偏差 53%、键角偏差 57°，健康结构在 ~1% / ~3°
以内）。

## 使用

挂钩在 `prepare_system` 尾部与每个 `min` leg 尾部，无需配置。Plan 的
`qc:` 段（全部可选）：

```yaml
qc:
  mode: soft                # soft（默认）仅报告；strict 在报告写出后抛
                            # StructureQualityError
  clash_heavy_nm: 0.2       # 重-重原子 clash 线（2.0 Å）
  clash_hydrogen_nm: 0.1    # 含 H 原子对（H-bond H...acceptor ~1.5 Å）
  bond_relative_tolerance: 0.25   # |r - r0| 相对偏差（下限 0.03 nm）
  bond_absolute_nm: 0.03
  angle_tolerance_deg: 30   # |theta - theta0|
  box_escape_fraction: 0.5  # 超出盒子半倍以上 = 体系破碎
```

`neomd validate` 用 collect-all 诊断（key path + did-you-mean）校验
`qc:` 段。

## 产物

- **`qc_report.json`**：每条 finding 携带原子索引、测量值、阈值；
  末尾是逐检查项 verdict + 总体 verdict（配体检查入 `ligand` 块）。

## 参考文献

- Isert, Atz, Schneider & Cremer, *Chem. Sci.* 15, 3670（2024）——
  [PoseBusters](https://doi.org/10.1039/D3SC04185H)（配体合理性检查的
  风格参照；RDKit 为 optional extra，不进核心依赖）。
- [OpenMM cookbook](https://openmm.org/) 的 clash/几何检查示例。
- [issue #15](https://github.com/NeoBinder/NeoDynamics/issues/15) ·
  [#7](https://github.com/NeoBinder/NeoDynamics/issues/7)（复现输入转
  QC 回归）。
