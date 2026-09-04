# 结构质量校验（neomd.qc）

> 状态：issue #15（含 #7 回归）· 已实现（`src/neomd/qc.py` + prepare/min 挂钩
> + `qc_report.json`）

## 背景与动机

issue #7（已关闭）报告：独立准备的蛋白与配体经 v1 `run_generic_md.py` 的
scipy-minimize 路径 min 后，局部原子出现不合理键长键角（scipy minimize 路径 +
约束处理相关 bug；OpenMM `minimizeEnergy()` 正常）。#7 的判定：
legacy 侧无独立工作项，**复现输入转为 #15 的 QC 回归用例**。

issue #15 的系统性动机：与其让用户"肉眼发现"破碎几何，不如让管线在
**建系后、min 后**自动执行结构质量校验（QC），把这类问题变成"管线自动拦截"。

## 与 issue 方案的差异

- **openmm-free 纯 numpy**：QC 模块为纯 numpy 几何计算（读 topology 文件坐标 +
  序列化 `system.xml` 的平衡值），**不经 kernel port**、不 import openmm —— 与
  #15 的判定一致（"QC 模块应为 openmm-free"）。
- **RDKit 设为 optional extra**：issue 中 PoseBusters 式立体化学检查依赖 RDKit，
  不进核心依赖；配体块检查在体系带 `input_files.ligands` 时自动运行，无配体则
  `skipped`（不是错误）。
- **fail-fast 改为 collect-all**：一次收集全部 findings 再统一裁决——strict 模式下
  报告写完后抛 `StructureQualityError`；默认 `soft` 仅报告（原始准备输入常带可修
  clash，min 正是解决它的手段）。
- 挂钩点：`prepare_system` 尾部（针对新写的 `solv.pdbx`/`system.xml`）与每个
  `min` leg 尾部（针对 min 后坐标 —— 正是 #7 记录的失效模式）；报告经 sinks 落盘，
  `md_run` 本身不写文件。

## 使用

README "Structure quality checks (`qc`)" 一节保留摘要与完整参数表。Plan 的
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

`qc_report.json`：每条 finding 携带原子索引、测量值、阈值；末尾是逐检查项 verdict
+ 总体 verdict。`neomd validate` 用常规 collect-all 诊断（key path +
did-you-mean）校验 `qc:` 段。

阈值用 #7 复现数据标定：其坏 minimize 留下键长偏差 53%、键角偏差 57°，而健康
minimized 结构在 ~1% / ~3° 以内；随附 fixture（3HTB smoke minimized、ala2
micro-fixture）零 finding 通过。

## 架构与产物

- 模块：`src/neomd/qc.py`，阈值与依据写在模块 docstring；检查项：NaN/Inf 坐标、
  出盒、PBC 感知最小镜像 clash（排除 1-2/1-3/1-4 键连对）、键长对 `r0` 偏离、
  键角对 `theta0` 偏离；体系带配体时同一套检查按配体范围再跑一遍，报告入
  `ligand` 块。
- 产物：`qc_report.json`（经 `ArtifactSink`）。
- 回归：`tests/v2/test_qc.py` —— issue #7 回归用例**由实际复现数据合成**
  （input / bad_min / good_min 三变体），加上公开接口测试。

## 参考文献与 ADR

- PoseBusters（*Chem Sci* 2024，配体合理性检查的风格参照）；OpenMM cookbook 的
  clash/几何检查示例。
- ADR-0001（strangler 迁移）；[issue #15](https://github.com/NeoBinder/NeoDynamics/issues/15)
  与 [#7](https://github.com/NeoBinder/NeoDynamics/issues/7)（复现输入转 QC 回归）。
