# ADR-0003：RBFE 引擎技术选型 —— openmmtools（prepare 边界依赖）而非整体 OpenFE 或全自研 softcore

- 状态：已确认（2026-09-03，Boresch restraint triple 出 ADR，RBFE 实装前生效）
- 决策者：项目维护者
- 关联：[issue #8](https://github.com/NeoBinder/NeoDynamics/issues/8)（RBFE）、
  [ADR-0001](0001-neomd2-strangler-migration.md)、AGENTS.md §Settled decisions
  （#5 双轨、#6 无永久兼容层、#8 多腿编排、#10 上游版本 pin 纪律）

## 背景

RBFE（relative binding free energy）引擎需要四块：

1. **alchemical 物理**：softcore 静电/色散形式、alchemical region 记账、
   λ 全局参数化 —— 微妙且必须经热力学检验的化学；
2. **λ 窗口编排**：N 个窗口各自成段动力学（settled decision #8 把多腿编排放到
   2.x，这里是第一次真实 revisit，见"后果"）；
3. **分析**：BAR/MBAR（`neomd.analysis` 的既有范围）；
4. **基准**：CDK2/trypsin 对拍文献。

v2 的架构约束（映射依据，均已核实）：

- **KernelPort 三适配器纪律**：openmm（生产）/ fake（毫秒级确定性 CI）/
  replay（金带回放）。fake 按 decision #9 只实现教科书物理，**不得**生长
  OpenMM corner-case 模仿。
- **alchemical 变换的 v2 先例已经存在**：barostat / `dummy_exceptions` /
  （ML/MM 计划中的）`ml_region` 都走同一条路 —— **System 形状的修改**，在
  Context 创建前由 openmm 适配器实现，经 `KernelSpec` 字段传入，fake 忽略。
  softcore 非键修改与它们同类（改的是 System 里的力，不是 bias）。
- **λ 在线可调的 seam 已存在**：`BiasParamOps.set_bias_param` 就是 v1
  `context.setParameter` 的 port 化；openmmtools 的 alchemical 力把 λ 暴露为
  Context 全局参数，同一机制直接可用。
- **卷首研究决定先例**（openmm-ml 评估，2026-09-02）：通用层可用就抽依赖，
  逐模型/逐方案的大块面则自研或 vendor（MIT + attribution）。

### 生态事实（2026-09 调研）

- **openmmtools 0.26.0**（conda-forge 2026-01-08，MIT）：
  `alchemy` 模块提供 `AlchemicalRegion` / `AlchemicalState` /
  `AbsoluteAlchemicalFactory` —— 对 OpenMM System 做 alchemical 手术的成熟
  实现（softcore 形式、λ 参数化、异常记账），是被 perses/OpenFE/TIES_MD 共同
  消费的底座。版本线 0.23→0.26 平稳，MIT，测试厚。
- **perses**（`HybridTopologyFactory`，单拓扑 RBFE 的参考实现）：最新
  release 0.10.3 约三年未动，事实休眠；开发重心已转移到 OpenFreeEnergy。
- **OpenFE**（v1.11.x，活跃，2025–2026 有 58 数据集 ~1700 变换的大规模协作
  基准）：**框架而非库** —— 自带 gufe 网络/计划模型、CLI、协议栈，其
  `RelativeHybridTopologyProtocol` 基于 perses 实现，生产栈实测为
  OpenFE 1.11 + OpenMM 8.4 + openmmtools 0.26。

## 决策

**RBFE 引擎以 openmmtools（`alchemy` 模块）为 alchemical 物理的
prepare 边界依赖；hybrid topology 构建以 vendor perses 派生代码（MIT，
attribution，逐字移植纪律）实现于 openmm 边界模块内；不整体引入 OpenFE
框架，不自研 softcore。** 具体：

1. **分层落点**：softcore/alchemical 力的生成是 **System 手术**，发生在
   prepare 侧（`tools/` 或 prepare.py 的 openmm 边界），产物是（每个变换
   一份）alchemical `system.xml` + λ 全局参数清单 —— 与 barostat /
   `dummy_exceptions` 完全同款的 `KernelSpec` 先例。**KernelPort 的 bias
   面（BiasIR）不新增 softcore 概念**：alchemical 修改是 system 形状，不是
   bias
   形状。openmmtools 因此**不在核心运行时 import 路径上**（同 openmm-ml
   决定里 openmm-ml 的降级逻辑：工具依赖，装了才用）。
2. **λ 阶梯**：同一份 alchemical system，各窗口不同的 λ 初值经
   Context 全局参数下置（`set_bias_param` 同一 seam 的
   `setParameter` 机制）；窗口间不共享 Context。
3. **窗口编排**：RBFE 实装自带一个**薄的** per-window 循环（N 次 `drive()`，
   共享 runner 级 manifest/账本），这是 decision #8 多腿编排的**受控最小
   revisit** —— 不是通用 `min → eq → prod` 管线（那仍是 2.x）。
4. **分析**：BAR/MBAR 落 `neomd.analysis`，读各窗口的
   `output.state`/λ-轨迹新带。
5. **基准**：CDK2/trypsin 对拍文献值（OpenFE 公开基准数据可直接作参照
   数字），这是物理正确性的唯一裁判（decision #9：金带只证行为不变）。
6. **版本纪律**：openmmtools pin 进 `pixi.toml`（decision #10 同款事件：
  显式 pin + 复核 `openmm_privates.py` 门 + 我们自身的 openmm 8.6.x pin
  与 openmmtools 的兼容矩阵核对）。

## 否决的替代方案

### 整体采用 OpenFE（框架捕获）

- OpenFE 自带计划/网络模型（gufe）、CLI 与整条协议栈；嵌入它等于把
  neomd 降级为其外挂驱动，KernelPort/fake/replay 纪律整体作废
- 双向耦合其发布节奏与其对 OpenMM 内部的使用方式（其协议直接操作
  OpenMM System/Context，无 port 可言）
- 违反"无永久兼容层"（#6）的精神：我们会永远维护一个 OpenFE 形状的垫片
- 它的 per-window 编排、restraint、分析全都自有版本，与我们的
  driver/probes/analysis 三套重叠

### 全自研 softcore（KernelPort 扩展）

- v1 无 softcore 先例（已核实），意味着**写新物理**而非逐字移植 —— 与
  settled decision #2 的精神（物理来自可查证的出处）相悖时需要极强的
  测试理由支撑
- softcore 静电/色散形式 + alchemical 异常记账 + 热力学一致性检验
  （ΔΔG 循环闭合、双精度数值路径）是 openmmtools 社区多年锤打过的微妙
  物理；自研等于重走这条路，且 fake kernel 无法复现其 corner-case
  （decision #9），测试只能全压在 openmm + 基准上
- 若走 KernelPort 新 bias kind 路线，还要动 port 面 —— 与"port 扩展集中
  管理"的冲突管控原则（计划 §四）不必要地相撞

## 关键配套决策

- **fake kernel 的角色**：alchemical system 对 fake 是不透明输入（它没有
  非键力可改），CI 用 mock λ-偏置（表格 bias 或简单表达式）驱动窗口编排/
  resume/带的确定性测试；物理 parity 由 openmm + 基准负责 —— 双轨规则
  （#5）的 alchemical 版
- **vendor 而非依赖 perses 包**：perses 休眠，依赖一个不动的包不如按
  openmm-ml 卷首决定同款纪律 vendor 其 hybrid topology 核心（MIT +
  attribution，逐字移植），让它接受我们的测试与 openmm 8.6 pin 约束
- **OpenFE 不进依赖，但作对拍参照**：基准数字、协议设置（λ schedule、
  softcore 参数缺省）从其文档/论文取用并引用

## 后果

### 正面

- 微妙物理买现成（openmmtools 0.26，MIT，测试厚），我们的工程量集中在
  v2 有优势的地方：编排、resume、产物、分析一体化
- KernelPort 面零扩张；alchemical 修改走既有 `KernelSpec` system-修改先例，
  架构叙事不变
- 解锁链清晰：`neomd.analysis`（BAR/MBAR）→ 本 ADR → λ 编排 spike
  （先 2 窗口 mock）→ 真实 softcore → CDK2/trypsin 基准

### 负面（已认领）

- openmmtools 的 OpenMM 版本跟随与我们 8.6.x pin 的兼容需要显式事件管理
  （#10 纪律承担）
- vendor 的 perses 派生代码成为我们的维护面（有界：hybrid topology 构建
  是一次性输入生成，不在运行路径）
- fake kernel 无法测试 softcore 本身 —— 物理正确性证据集中在 openmm 侧
  基准，CI 绿不等于物理对（#9 的既认代价在 RBFE 上放大）

## 重开条件

- openmmtools 停滞或与我们的 OpenMM pin 不可调和 → 升级为自研 softcore
  的 port 扩展 ADR（届时 softcore 物理已有 vendor 参照可逐字移植）
- 需要 openmmtools 不提供的 softcore 形式/增强采样耦合（如与 OPES/GaMD
  的联合）→ 同上
- 若 OpenFE 未来提供可嵌入的纯库形态（无 gufe/CLI 耦合）且维护活跃，
  可重新评估替代 vendor 代码
