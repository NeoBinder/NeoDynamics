# RBFE（相对结合自由能，λ 窗口）

> 状态：issue #8（主切片）· 实现状态：已实装（`methods/rbfe.py` +
> `neomd/rbfe.py::run_ladder` + `neomd.analysis.freeenergy`，fake 确定性 CI 全绿，
> openmm 侧 openmmtools 门控冒烟在 `rbfe` pixi env）·
> ADR：[ADR-0003](../adr/0003-rbfe-technology-selection.md)（选型）、
> [ADR-0007](../adr/0007-rbfe-lambda-window-orchestration.md)（λ 编排 / ParamEnergy / du.tsv）
> · 姊妹文档：[boresch.md](boresch.md)（RBFE 的标准锚 restraint）

## 背景与动机

issue #8 的出发点：公司核心业务（蛋白设计 / 酶-底物优化）需要对**突变体×底物
系列**做定量亲和力/选择性排序，RBFE 是当前精度-成本最优的定量方法（公开基准
RMSE ~1.0–1.3 kcal/mol，中性边 2–12 GPU·h）。issue 设定的闭环是：
扰动构建 → λ 窗口调度 → 采样 → BAR/MBAR 分析。

issue 写作时基于 v1 代码盘点基础设施（`builder`、约束构造、`Pipeline`
复用、`bin/` 分析脚本）；这些职责现已分别落在 `tools/`、restraint
triple（见 [boresch.md](boresch.md)）、`drive()`/`PreparedMethod` 与
`neomd analysis` CLI。本文只描述当前实装形状。

## 与 issue 方案的差异

issue #8 的技术方案给出两条扰动层路线（A：openmmtools/OpenFE 薄封装；
B：自研轻量 softcore）。取舍记录在 ADR-0003/0007：

- **扰动层（ADR-0003）**：选 openmmtools `alchemy` 模块做 **prepare 边界
  依赖**——softcore/alchemical 力的生成是 System 手术（与 barostat /
  `dummy_exceptions` 同款 `KernelSpec` 先例），**不整体引入 OpenFE 框架，
  也不自研 softcore**；hybrid topology 构建以 vendor perses 派生代码
  （MIT + attribution，逐字移植）实现。openmmtools 只装在**独立的 `rbfe`
  pixi env**，永不进默认依赖与核心运行时 import 路径。
- **λ 载体（ADR-0007）**：窗口 λ 经 `KernelSpec.global_parameters` 在
  Context 创建时下置（openmm 侧即 openmmtools 的
  `lambda_electrostatics`/`lambda_sterics`）；BiasIR 不新增 softcore 概念
  ——alchemical 是 system 形状，不是 bias 形状。
- **λ 窗口编排 = 多腿决策 #8 的第一个受控 revisit（ADR-0007）**：
  settled decision #8 把通用 `min→eq→prod` 多腿编排推到 2.x；
  `neomd.rbfe.run_ladder` 是刻意收窄的最小 revisit——每窗一次完整
  `drive()`、runner 级 `ladder.json` 账本、中断窗口自动续跑；不做腿间
  传递、通用 DAG、跨窗并行。
- **fake 路径**：fake kernel 没有非键物理可 alchemify，窗口 plan 携带
  `alchemical.mock_bias`（λ 缩放的谐距离偏置），让编排/resume/du 带/
  BAR-MBAR 管线在 CI 里确定性可测；softcore 物理正确性由 openmm 侧
  基准负责（decision #9 的 alchemical 版代价）。
- **分析（ADR-0007 §5）**：BAR/MBAR 自实现（numpy-only，logaddexp 稳定，
  解析谐振子基准钉住），pymbar 只作装了才跑的 gated 对拍，不进默认依赖。
- **λ 排布**：issue 提议的"自适应 λ 分配（省 >85% 算力）"不在本期；
  当前 ladder 为固定列表（初版 11–24 窗的建议仍适用）。

## 使用

以下内容迁移自 README 并扩充；README 保留摘要与链接。

### 单窗 plan（fake kernel 可跑的最小示例）

```yaml
method: rbfe
steps: 50000
temperature: 298
seed: 2026
integrator: {dt: 0.002, friction_coeff: 1.0}
input_files:
  complex: /work_dir/min/last.pdbx
  system: /work_dir/sys_prep/htb/system.xml
output:
  output_dir: /work_dir/rbfe/win_00   # one directory per window
  report_interval: 500                # du.tsv row stride
alchemical:
  lambda_values: {lambda_alchemical: 0.0}   # THIS window's λ
  ladder:                                    # every window, in order
    - {lambda_alchemical: 0.0}
    - {lambda_alchemical: 0.5}
    - {lambda_alchemical: 1.0}
  mock_bias: {grp1_idx: "0", grp2_idx: "1", # fake-kernel test potential
              k_kj_mol_nm2: 50.0, r0_nm: 0.3}
```

要点：`lambda_values` 是**本窗**的 λ（必须取自 `ladder` 的某一项，
`plan.py` collect-all 校验）；`ladder` 是全部窗口的 λ 顺序表。示例是
fake 档的 3 点 mock ladder；openmm 档上 `mock_bias` 让位于真实
alchemical `lambda_values`（openmmtools 构建的 system 经
`KernelSpec.global_parameters` 下置 λ）。

### run_ladder（整条阶梯）

```python
from neomd.rbfe import run_ladder
outcome = run_ladder(plan)   # N windows, ladder.json ledger, auto-resume
```

输入一份 `method: "rbfe"` 的 Plan（含 `alchemical.ladder`），输出
`window_00…` N 个窗口目录（每窗一次完整 `drive()`：自己的 manifest、
checkpoint、du 带）+ runner 级账本 `ladder.json`。每窗 seed = 基础
seed + 窗序（独立链）；中断续跑：manifest 存在且末 epoch 不是
`done:rbfe` 的窗口自动置 `continue_md`，交给唯一 resume 所有者。

### analysis：BAR / MBAR

```bash
neomd analysis bar  /work_dir/rbfe/win_00 /work_dir/rbfe/win_01
neomd analysis mbar /work_dir/rbfe/win_00 /work_dir/rbfe/win_01 /work_dir/rbfe/win_02
```

BAR 取两窗、MBAR 取整条阶梯，输入就是窗口目录列表（du 带自描述，
读回即重建 ladder）。

## 架构与产物

- **ParamEnergy**：KernelPort 第四个可选能力协议
  （`energy_with_params(params) -> float`）——当前构型在**临时**下置的
  全局参数下的总势能（kJ/mol），不下步、不扰动动力学状态、返回前恢复
  每一个被触碰的参数。openmm 与 fake 实现，replay 不提供。
- **du.tsv**：一个 λ 窗口一条 du 带（`DuProbe`，registry preset `"du"`）：
  每行一个观测步、每列一个 ladder 档位（`u_000…`，kJ/mol，经
  ParamEnergy 评出）；**λ 参数注释行**（`# lambda_sterics <v> <v> …`）
  使带自描述。势能含全部项（alchemical、boresch 锚……）——跨 λ 恒定的
  部分在 BAR/MBAR 估计量里精确相消。resume 时按 tsv 通用规则截断
  （`resume.py` 唯一所有者）。
- **ladder.json**：runner 级账本——ladder、每窗 λ、du 带末步、结果摘要。
- **resume**：窗口目录的 manifest + epoch 链（`resume:<step>`）驱动
  自动续跑；编排器自己不碰带。
- **环境**：openmmtools 只在 `rbfe` pixi env（prepare 边界依赖），
  版本 pin 进 `pixi.toml`（decision #10 纪律）；该 env 的门是
  `pixi run -e rbfe test-rbfe`（默认档 + openmmtools 门控 alchemy 冒烟）。

## 参考文献与 ADR

- [ADR-0003：RBFE 引擎技术选型](../adr/0003-rbfe-technology-selection.md)
- [ADR-0007：λ 窗口编排、ParamEnergy port 能力与 du.tsv 带](../adr/0007-rbfe-lambda-window-orchestration.md)
- Boresch, Karplus et al., *Absolute Binding Free Energies: A Quantitative
  Approach*, J. Phys. Chem. B 2003, 107, 9535–9551（锚 restraint，见
  [boresch.md](boresch.md)）
- OpenFE 工业基准（ChemRxiv 2025：中性边 5.8±3.4 GPU·h @ L40S）；
  LiveCoMS RBFE best practices（issue #8 引用的公开基准来源，作
  对拍参照，不作依赖）
- Bennett 1976（BAR）；Shirts & Chodera 2008（MBAR）
