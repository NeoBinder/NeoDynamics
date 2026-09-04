# ADR-0007：RBFE λ 窗口编排、ParamEnergy port 能力与 du.tsv 带

- 状态：已确认（2026-09-10，RBFE 实装时随代码收敛）
- 决策者：项目维护者
- 关联：[ADR-0003](0003-rbfe-technology-selection.md)（RBFE 选型，本文的母决策）、
  [issue #8](https://github.com/NeoBinder/NeoDynamics/issues/8)、
  AGENTS.md §Settled decisions（#5 双轨、#8 多腿编排、#9 fake 不证物理）
- 前置：`neomd.analysis`、Boresch restraint triple

## 背景

ADR-0003 定了 RBFE 的选型骨架（openmmtools 走 prepare 边界、λ 走
Context 全局参数、BAR/MBAR 落 analysis）。RBFE 实装时剩下三个必须落纸的
接口决策，本文即它们：

1. du 值（同一构型在相邻 λ 下的势能）怎么从 kernel 拿出来；
2. du 带的 artifact 形状（BAR/MBAR 的输入，要能 resume）；
3. λ 阶梯（N 个窗口）怎么编——多腿编排（settled decision #8）的第一个
   真实 revisit 该长什么样。

## 决策

### 1. `ParamEnergy` —— 可选 port 能力，不是 bias 面

`KernelPort` 新增第四个可选能力协议 `ParamEnergy`
（`energy_with_params(params) -> float`）：当前构型在**临时**下置的
全局参数下的总势能（kJ/mol），不下步、不扰动动力学状态、返回前恢复
每一个被触碰的参数。

- 它是 "setParameter + getState(getEnergy) + setParameter 回去" 的
  port 化拼写。做成能力（而不是 probe 侧借 `BiasParamOps` 推拉）的理由：
  save/restore 配对必须原子，且 du probe 不该知道任何 kernel 细节。
- **BiasIR 不新增 softcore 概念**（ADR-0003 第 1 条的重申）：alchemical
  是 system 形状，经 `KernelSpec.global_parameters`（新 spec 字段，
  barostat/dummy_exceptions 同款先例）在 Context 创建时下置；resume 时
  checkpoint 携带的参数值覆盖它（checkpoint 是在该窗口自己的 λ 下写的）。
- openmm 与 fake 两个适配器实现它（确定性）；replay 不提供（回放带没有
  参数）。openmmtools 的 `lambda_electrostatics` / `lambda_sterics` 和
  fake 的 mock `lambda_alchemical` 走同一个能力，probe 不区分。
- source-scan 不需要扩：能力协议经 `provides()` 协商，无 reach-through。

### 2. `du.tsv` —— 自描述、可 resume 的新 v2 artifact

一个 λ 窗口一条 du 带（`DuProbe`，registry preset `"du"`），每行一个
观测步、每列一个 ladder 档位（`u_000…`，kJ/mol，经 `ParamEnergy` 评出）。
**λ 参数注释行**（`# lambda_sterics <v> <v> …`，每参数一行、每列一值）
让带自描述——读回即可重建整个 ladder（`neomd.analysis.freeenergy.read_du`）。
注释行天然在 resume 截断后存活，`resume.py` 按 tsv 通用规则截断，与
colvar/smd 带同一所有者。势能含全部项（alchemical、boresch 锚……）——
对每个样本跨 λ 恒定的部分在 BAR/MBAR 估计量里精确相消。

### 3. `neomd.rbfe.run_ladder` —— 受控的薄编排，不是通用多腿

settled decision #8（min→eq→prod 推到 2.x）的**最小受控 revisit**，
形状刻意收窄：

- 输入一个 `method: "rbfe"` 的 Plan（含 `alchemical.ladder`），输出
  N 个窗口目录（`window_00…`，每个是一次完整 `drive()`：自己的 manifest、
  checkpoint、du 带）+ 一份 runner 级账本 `ladder.json`（ladder、每窗 λ、
  du 带末步、结果摘要）。
- 每窗 seed = 基础 seed + 窗序（独立链）；窗口 λ 经
  `alchemical.lambda_values` 覆写进窗口 plan（必须在 ladder 内——
  `plan.py` collect-all 校验）。
- **中断续跑**：窗口目录的 manifest 存在且末 epoch 不是 `done:rbfe` 的
  窗口自动置 `continue_md`，交给唯一 resume 所有者；编排器自己不碰带。
- 不做：腿间传递（窗口各自从同一初始构型出发）、通用 DAG、跨窗并行。
  这些仍属 2.x 多腿编排决策；`run_ladder` 是它的第一个客户与探针。

### 4. fake 路径 —— mock λ-偏置（ADR-0003 认领方案的落地形状）

fake kernel 没有非键物理可 alchemify，窗口 plan 携带
`alchemical.mock_bias`（两原子组 + 力常数 + 平衡距离）：
`lambda_alchemical*(k/2)*(d - r0)^2`，λ 是该窗口参数值。
CI 用它证明编排/resume/du 带/BAR-MBAR 管线的确定性；softcore 物理由
openmm + 基准负责（decision #9 的 alchemical 版代价）。

### 5. 分析 —— BAR/MBAR 双估计器，numpy-only，pymbar 只作 gated 对拍

`neomd.analysis.freeenergy`：BAR（Bennett 方程 = MBAR K=2 的驻点条件，
bracket + 二分求根，全 logaddexp 稳定）与 MBAR（自洽迭代 + 每样本
own-state 能量平移稳定化）。解析基准（1-D 谐振子阱闭式
`ΔF = ½ ln(k₁/k₀)`）钉住两者。pymbar 装了才跑的对拍测试留在
openmmtools 门控层，永不进默认依赖。

## 否决的替代方案

- **du 经 BiasParamOps 在 probe 侧推拉**：save/restore 配对被拆到调用方，
  任一异常路径都可能把动力学留在错误的 λ 下；能力协议一次封装。
- **du 带记录约化功（β·Δu）而非势能**：带会锁死温度，重分析/变温不可
  能；存 kJ/mol + manifest 温度是更晚的绑定。
- **N 窗口塞进一次 drive()（一个 Context 换 λ 续跑）**：窗口间系综
  独立性被静默破坏（热力学积分要求每窗独立采样），且 resume 语义
  （checkpoint 在哪个 λ 下写的）会变含糊。每窗一次 drive() 是对的。
- **通用多腿编排现在就做**：#8 明确推后；RBFE 只需要一个受控循环。

## 后果

- 正面：KernelPort 面只加一个最小可选能力 + 一个 spec 字段，三适配器
  纪律不破；du 带自描述使 `analysis bar/mbar` 只需窗口目录列表；
  `run_ladder` 为 2.x 多腿编排提供第一个真实用例的经验数据。
- 负面（已认领）：fake 档 CI 绿不证 softcore 物理（#9 代价在 RBFE 放大，
  ADR-0003 已认领）；`ParamEnergy` 在 openmm 侧每次观测做 K 次
  re-parameterize + 能量读，K 大时有真实开销（可后续在 du 带节流，
  不在本 ADR 范围）。

## 重开条件

- 2.x 通用多腿编排落地时，`run_ladder` 应重写为其一个客户；
- 若 du 采样成为瓶颈，考虑 kernel 侧批量邻近-λ 读取（一个能力方法
  返回整行），届时扩展本 ADR。
