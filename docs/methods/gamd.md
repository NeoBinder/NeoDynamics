# GaMD（method: gamd）

> **来源 issue**：[#10 [Feature] 实现 GaMD（Gaussian accelerated MD）管线](https://github.com/NeoBinder/NeoDynamics/issues/10)
> **实现状态**：已实现（`src/neomd/methods/gamd.py`，真实插件）
> **关联 ADR**：[ADR-0005 GaMD boost 内核缝](../adr/0005-gamd-boost-seam.md)
> （另涉 [ADR-0002 plugin plan-schema namespace](../adr/0002-plugin-plan-schema-namespace.md)）

## 背景与动机

GaMD（Gaussian accelerated MD，Miao et al., *JCTC* 2015）通过对势能面加
谐波 boost ΔV(P) = ½·k·(E−P)²（P < E 时，否则 0）实现**无 CV** 增强采样，
适合没有明确 CV 的场景（口袋开合、loop 运动、配体进出路径探索），与
metadynamics 形成互补。LiGaMD（ligand GaMD，Miao, *JCP* 2020）专门加速
配体结合/解离，对酶-底物体系尤其相关（trypsin-benzamidine ΔG 与实验吻合）。

issue #10 的原始方案基于 v1 结构（独立 `neomd.gamd` 子包 + `bin/run_gamd.py`
入口）；当前落点：

- **`src/neomd/methods/gamd.py`**：方法知识三元组，经 registry 分发、
  prepare 契约调度，无独立子包与 `bin/` 入口；
- boost 标定用 `port.energy_forces()` / `group_energy()`（既有 port 操作）；
  LiGaMD 组 boost 用 `GroupEnergy` + `pick_free_force_group`；参数在线更新
  用 `BiasParamOps` 同款 seam（`set_boost_param`）；
- reweighting 依赖 issue #16 的 `neomd.analysis` 子包（w = exp(βΔV)）。

## 与 issue 方案的差异

issue 给出两个候选实现（复用 gamd-openmm vs 自研）。决策（ADR-0005，
2026-09-03）：**BoostOps 自研 seam；弃用 gamd-openmm 的乘性双 boost 缩放**。

- **BoostOps 能力协议**（与 BiasOps / BiasParamOps / GroupEnergy 并排，
  `provides()` 协商）：`install_boost(channels)`（预 Context 安装）、
  `set_boost_param(label, name, value)`（在线推 threshold/k）、
  `boost_potentials()`（读最近一步 ΔV / P / s）。原因：GaMD 的偏置力是
  **缩放的系统力** F\* = −(1−k(E−P))·∇P，BiasIR 加性偏置在数学上表达不出
  ΔV(P(x))（其力来自自身表达式，不乘其它 force group 的力）。
- **加性多通道缩放，放弃 gamd-openmm 的乘性 s_P·s_D**：多通道
  （dual boost、LiGaMD 叠加）按每 group 加性 s(g) = 1 + Σ_c ΔV_c′(P_c)，
  这是修改势 V\* = Σ_g V_g + Σ_c ΔV_c(P_c) 的**精确梯度**；gamd-openmm 的
  s_P·s_D 不是任何势的梯度，采样分布与上报的 ΔV 不再出自同一个 V\*，
  重加权自洽性被破坏，故弃用（ADR-0005"否决的替代方案"）。
- **标定预跑是方法侧纯逻辑，不是内核缝**（否决积分器内 Welford 窗口）：
  零强度安装 → 无偏标定预跑（numpy 统计 Vmax/Vmin/σV，选文献两模式的
  (E, k)）→ 经 `set_boost_param` 在线推参；标定数学有纯函数定义点，
  测试可解析对拍。
- **resume 从 `gamd_calibration.json` 读回参数重推，不再标定**（推送幂等，
  checkpoint 本身带积分器变量），恢复路径与直跑共享同一参数来源。
- 其余 issue 任务项（reweight 不确定度输出、配体结合示例）为后续项。

## 使用

GaMD 换入 `method: gamd` 和一个 `gamd:` 段：`mode: total` 或 `dual`
（total + 二面角通道，扭转力自动隔离进独立 force group）；显式
`channels: [{label, groups}]` 定义覆盖 LiGaMD 式体系（系统 XML 中那些
相互作用已自成 force group）；`sigma0`（kJ/mol，默认 6.0）为 boost 强度
旋钮；标定预跑长度/间隔在 `calibration_steps` / `calibration_interval`
——boost 安装时零强度，`steps` 内先跑一小段无偏标定（`steps` 是**最终
步数**），按文献阈值/谐波对选出参数、写 `gamd_calibration.json` 并在线
推送。最小可运行 plan（total boost；restraint 墙只是给标定一点势能起伏）：

```yaml
method: gamd
steps: 500000              # 最终步数 —— 标定在其内部跑
temperature: 298
seed: 2026
integrator: {dt: 0.002, friction_coeff: 1.0}
input_files:
  complex: /work_dir/min/last.pdbx
  system: /work_dir/sys_prep/htb/system.xml
output:
  output_dir: /work_dir/gamd
  trajectory_interval: 1000
  checkpoint_interval: 1000
  report_gamd: true        # 写 gamd.tsv（默认开）
restraint:                 # 可选，建议标定前加
  dist: {type: distance, grp1: "0", grp2: "1", restr_k: 500.0, max_nm: 0.8}
gamd:
  mode: total              # 或 dual（total + 二面角通道）
  sigma0: 6.0              # kJ/mol，boost 强度旋钮
  calibration_steps: 50000 # 无偏预跑，含在 steps 内
  calibration_interval: 50 # Vmax/Vmin/σV 采样步长
  frequency: 10            # gamd.tsv 记录步长
  # LiGaMD 式：对预先分好的 force group 显式定义通道：
  # mode: channels
  # channels: [{label: ligand, groups: [3]}]
```

## 架构与产物

- **方法 triple**：`methods/gamd.py` 注册为 `method: gamd`；prepare 时
  以零强度通道 `install_boost`（晚于全部 `install_bias`——安装后再
  install_bias 直接报错，GaMD 不与 metadynamics 组合），随后方法侧
  标定 → `set_boost_param` 推 (threshold, k)。
- **内核落点**（ADR-0005）：openmm 适配器构造 boost 版 Langevin
  CustomIntegrator（更新形式自 gamd-openmm 移植）替换尚未建 Context 的
  积分器；fake 适配器在 numpy Langevin 循环里做同一数学（boost 模式下
  传播自身势的数值梯度力，按通道缩放进 Euler-Maruyama；未装 boost 时
  逐位不变）。dual boost 的二面角组发现走 duck-typed 伴随方法
  `torsion_force_groups()`（组 id 保持不透明）。
- **tape / artifact**：`gamd.tsv`（GamdProbe，每通道 ΔV / 目标能量 P /
  力缩放 s——重加权 trace；switch `output.report_gamd`，resume 时照常
  裁剪）；`gamd_calibration.json`（每通道 Vmax/Vmin/σV 样本与选定的
  (threshold, k)——唯一参数来源）。
- **resume 语义**：`continue_md: true` 不重新标定——`resume.py` 裁剪
  `gamd.tsv` 至 checkpoint 步，恢复时从 `gamd_calibration.json` 读回参数
  重推（幂等）。步数记账为绝对步（标定 + 生产共享同一坐标系）。
- **与 analysis 的衔接**：重加权 `w = exp(βΔV)` 走 `neomd.analysis`
  （`neomd analysis reweight`，读运行产物格式）；不确定度输出为
  后续项。

## 参考文献

- Miao, Feher & McCammon, *JCTC* 11, 5208（2015）—— GaMD（issue 引用）。
- Miao, *JCP* 2020 —— LiGaMD（issue 引用，trypsin-benzamidine ΔG 与
  实验吻合）；dual-boost 与阈值两模式另见 Miao 2016。
- Copeland, Miao 等, *JPCB* 2022 —— gamd-openmm（ADR-0005 引用；
  其 Langevin 更新形式被移植，乘性双 boost 缩放被否决）。
- [ADR-0005](../adr/0005-gamd-boost-seam.md) —— BoostOps 决策全文
  （含全部否决的替代方案）。
- [issue #10](https://github.com/NeoBinder/NeoDynamics/issues/10) ——
  原始需求与 v2 适配判定。
