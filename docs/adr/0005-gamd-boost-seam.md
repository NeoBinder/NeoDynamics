# ADR-0005：GaMD boost 内核缝 —— port 新增 BoostOps 能力（install_boost / set_boost_param / boost_potentials），而非 BiasIR 加性偏置或 KernelSpec 字段

- 状态：已确认（2026-09-03，Wave 2 轨道 W2-b）
- 决策者：项目维护者
- 关联：[ADR-0002](0002-plugin-plan-schema-namespace.md)、[issue 开发计划 W2-b](../issue-dev-plan.md)（issue #10）、GaMD 文献（Miao/Feher/McCammon, JCTC 2015；Miao 2016 dual-boost；Miao/Bhattarai/Wang, JCTC 2020 LiGaMD；Copeland/Miao 等, JPCB 2022 gamd-openmm）

## 背景

GaMD 的 boost 势 ΔV(P) = ½·k·(E−P)²（P < E 时，否则 0）依赖体系**自身**的
势能 P，因此偏置后的力是**缩放的系统力**：F\* = −(1+ΔV′(P))·∇P =
−(1 − k(E−P))·∇P。一个加性偏置力（BiasIR → install_bias）表达不出它：
BiasIR 的力来自其自身表达式，不乘其它 force group 的力。既有 OpenMM 路线
（gamd-openmm、AMBER、NAMD）都是 CustomIntegrator 每步把各 force group 的
能量（`energy`、`energy1`…）读进全局变量，算出 ΔV 与缩放因子，再在
Langevin 更新里逐 group 缩放力。v2 需要在 KernelPort 缝上等价的东西——
这是 issue 开发计划点名的 "port 扩展" 批次的一部分。

## 决策

**KernelPort 新增一个可选能力协议 `BoostOps`（与 BiasOps / BiasParamOps /
GroupEnergy 并排，`provides()` 协商）：**

```python
install_boost(channels: Sequence[BoostChannelIR]) -> None   # 预 Context 安装
set_boost_param(label, name, value) -> None                 # name ∈ {"threshold", "k"}
boost_potentials() -> dict[label, BoostReading]             # 最近一步的 ΔV / P / s
```

`BoostChannelIR(label, groups, threshold, k)`：一个 boost 通道 = 一组 force
group 的能量之和作为目标 P（`groups == ()` 是 "total" 通道：除 install_bias
分配的偏置组外全部系统组——加性偏置不改物理体系），`threshold` = E，
`k` = 有效谐波常数（k = k0/(Vmax−Vmin)，0 < k0 ≤ 1，保证缩放因子
s = 1 − k(E−P) ∈ [0,1]、力永不反向）。多通道（dual boost、LiGaMD 叠加）
按**每 group 加性**缩放：s(g) = 1 + Σ_{c∋g} ΔV_c′(P_c) —— 这是修改势
V\*(x) = Σ_g V_g + Σ_c ΔV_c(P_c) 的**精确梯度**；gamd-openmm 的双 boost
用 s_P·s_D 乘性缩放，那不是任何势的梯度（重加权一致性要求力与 ΔV 出自
同一个 V\*，故弃用，见"否决的替代方案"）。ΔV_trace 由
`boost_potentials()` 报告（openmm：积分器全局变量；fake：numpy 循环里
闭式计算），GaMD 插件的 `gamd.tsv` 录带经它读取。

**适配器落点：**

- `kernel/openmm.py`：`install_boost` 以零强度通道（k=0, E=1e99）构造一个
  boost 版 Langevin CustomIntegrator（Langevin 更新形式自 gamd-openmm 移植：
  vscale/fscale/noisescale 常数 + 每通道 dV/b 缩放计算 + 逐 force group 的
  `fscale*f{g}*(1 − Σ b_c)/m` 更新），**替换尚未建 Context 的
  `self._integrator`**——与 install_bias 同一"预 Context 纪律"；Context 已
  存在则报错（Context 的积分器构造后不可换）。标定/恢复参数经
  `set_boost_param`（`integrator.setGlobalVariableByName`）在线推送，无
  reinitialize。**未安装 boost 时 `_make_integrator` 一字不改**，非 boost
  路径（golden tapes、全部既有 CI）不受影响。
- `kernel/fake.py`：numpy Langevin 循环里做同一数学。boost 安装后 fake
  开始传播自身（几何偏置）势的力（`_numerical_gradient` 机器已有，按
  force group 分组求梯度），按通道缩放后进入 Euler-Maruyama 更新；未安装
  boost 时 fake 动力学与快照格式**逐位不变**（文档化的 F=0 简化只在无
  boost 时成立）。snapshot/restore 携带通道状态，中断恢复后轨迹逐位一致。
- 安装顺序约束：**install_boost 必须晚于全部 install_bias**（drive() 的
  天然顺序：restraint 安装先于方法 prepare），之后 install_bias 直接报错
  ——openmm 的逐 group 更新串在安装时固化，静默忽略新组是物理 bug，宁可
  大声失败。两个适配器同一守卫。

**标定预跑是方法侧（插件）逻辑，不是内核缝**：标定只是无偏 MD 采样 +
每 `calibration_interval` 步读 `energy_forces()` / `group_energy()`（既有
port 操作），统计 Vmax/Vmin/Vavg/σV，选 (E, k)（文献两模式：下界
E = Vmax, k0 = min(1, (σ0/σV)(Vmax−Vmin)/(Vmax−Vavg))；上界
k0 = (1−σ0/σV)(Vmax−Vmin)/(Vavg−Vmin)，0<k0≤1 可用否则回落下界，
E = Vmin + (Vmax−Vmin)/k0）。零强度安装 → 标定推参 → 生产，这条顺序同时
解决了 resume：恢复时跳过标定，从 `gamd_calibration.json` 读回参数推送
（checkpoint 本身带积分器变量，推送幂等），恢复路径与直跑路径共享同一
定义点。

## 否决的替代方案

### 把 boost 写成 BiasIR 加性偏置

数学上不可能：ΔV(P) 是 P 的函数，其力是 ∇P 的标量倍；BiasIR 的力来自
自身表达式，不依赖其它组的能量。加性偏置只能表达 ΔV(x)（几何函数），
不能表达 ΔV(P(x))。

### `KernelSpec.boost` 字段（barostat 先例）

boost 通道的 (E, k) 来自标定预跑，构造 kernel 时未知；且 drive() 在方法
分发**之前**建 kernel，而 core 不读 `plugins.*` 段（ADR-0002：段内容对
核心不透明）。KernelSpec 字段会迫使 core 解释插件配置。零强度安装 +
`set_boost_param` 在线推参绕开整个问题（BiasParamOps 的同款形状）。

### 在活 Context 上换积分器（mid-run swap）

Context 的积分器构造后不可换；重建 Simulation + State 搬运会重置噪声流，
且引入 openmm Privates 风险。标定改为跑在零强度 boost 积分器上（反正
GaMD 的 cMD 相没有位串一致性包袱），swap 就完全不需要了。

### gamd-openmm 的双 boost 乘性缩放（s_P·s_D）

不是任何势的梯度（∇×(s_P s_D F_D) ≠ 0），采样分布与上报的
ΔV = ΔV_P + ΔV_D 不再来自同一个 V\*，重加权自洽性被破坏。加性形式
s(g) = 1 + Σ_c ΔV_c′ 是 V\* 的精确梯度，代价只是每 group 更新串里
多几个加项。

### 内核侧自动标定（gamd-openmm 的积分器内 Welford 窗口）

积分器内做统计要求把 (Vmax, Vmin, σV) 状态放进积分器全局变量并用
CustomIntegrator 表达式实现 Welford——可测试性差（fake 与 openmm 各写
一遍易漂移的浮点串）、参数选择不可审计。方法侧（插件）numpy 标定 +
`set_boost_param` 推参让标定数学有一个纯函数定义点，测试可解析对拍。

## 后果

### 正面

- 采样方法获得第三类内核能力（加性偏置 → 表格偏置 → **能量依赖的力缩放**），
  BoostOps 协议一次定义、双适配器实现，`provides()` 协商；
- 未安装 boost 的路径（golden、全部既有测试）零改动；
- 标定/阈值选择是插件的纯函数，closed-form 可测；resume 与直跑共享同一
  参数来源（gamd_calibration.json + checkpoint 里的积分器变量）。

### 负面（已认领）

- fake 内核在 boost 模式传播数值梯度力：每步 O(组数 × 3N) 次能量求值，
  fake 只服务小体系短测试（毫秒级），可接受；
- install_boost 后禁止再 install_bias：GaMD 不与 metadynamics 组合（本期
  范围外），守卫把组合需求显式挡下而不是静默错算；
- boost 路径的 Langevin 更新形式（gamd-openmm 移植）与 openmm
  LangevinIntegrator 不逐位等价——GaMD 是新物理，无 golden 基线可破，
  非等价性文档化即可。

## 实现补记（W2-b 落地时）

dual boost 的二面角通道需要"哪些 force group 持有扭转能"这一发现能力。
落在 BoostOps 之外的一个 duck-typed 伴随方法
`torsion_force_groups() -> tuple[int, ...]`（与 GroupEnergy 同款协商风格，
port.py 文档化）：openmm 端返回/隔离（pre-Context `setForceGroup` +
`pick_free_force_group`）体系扭转力；fake 端返回已安装的二面角偏置的
组。组 id 保持不透明，只被原样交回 `install_boost`。LiGaMD 场景不自动
猜测配体分组：显式 `gamd.channels: [{label, groups}]` 指向系统 XML 中
已自成 force group 的配体扭转/非键相互作用。

另：openmm 的 `step(x)` 在 x=0 处不为 0，不能用于"dV 在 P≥E 时归零"；
dV 表达式因此写为 `0.5*(E-P)*b`（未钳位区间内恰为 ½k(E−P)²，b=0 时
恰为 0，仅在越界钳位 b=1 的数值防护区内偏离 ½k(E−P)²）。

## 重开条件

出现第二个能量依赖力缩放的方法（或需要 GaMD × metadynamics 组合）时，
重审"安装后禁止 install_bias"守卫与逐 group 更新串的固化方式；ML/MM
（#12 `KernelSpec.ml_region`）落地时若也需积分器协作，并入同一批
port 扩展审阅。
