# GaMD（method: gamd）

> 需求与设计决策：[issue #10](https://github.com/NeoBinder/NeoDynamics/issues/10) ·
> [ADR-0005 GaMD boost 内核缝](../adr/0005-gamd-boost-seam.md)（BoostOps seam、
> 加性多通道缩放、全部被否决的替代方案）

## 原理简述

GaMD（Gaussian accelerated MD）对势能面加谐波 boost
ΔV(P) = ½·k·(E−P)²（0 < k(E−P) ≤ 1 时，否则采用下面的连续延拓），boost 力是缩放的系统力
F\* = −(1−k(E−P))·∇P，实现**无 CV** 增强采样——适合没有明确 CV 的场景
（口袋开合、loop 运动、配体进出路径），与 metadynamics 互补。
LiGaMD 对配体相关 force group 额外加一个通道，专门加速结合/解离。
 boost 参数 (E, k) 由无偏标定预跑按文献阈值/谐波两模式自动选出，
逐帧精确指数权重可作诊断；生产 PMF 使用分箱后的二阶累积量展开（CE2）。

越出标定区间且 k(E−P) > 1 时，ΔV 延拓为 `(E−P)−1/(2k)`，
与钳位后的力缩放保持同一势能的导数；k=0 时 boost 为零。
OpenMM 将完整 boost 写入 System 势能，NPT 的候选箱求值会同时
评估基础势、约束和所有启用的 boost。MetaD 独立模式中的已累积
偏置同样由其 System Force 计入，不能重复相加。参见
[ADR 0009](../adr/0009-ordered-potential-boosts.md)。

为精确恢复 NPT，内核管理各向同性体积提议：检查点保存随机数状态、
提议宽度和接受计数。OpenMM 8.6 原生检查点遗漏这些 barostat 内部状态，
不能据此声称逐步精确恢复。新的完整状态格式对调用者仍是 opaque checkpoint。

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

GaMD 不与 metadynamics 组合（安装 boost 后再 `install_bias` 直接报错）。

## 产物

- **`gamd.tsv`**：保存构型处的每通道 ΔV / 原始能量 P / 力缩放 s，
  以及 `total_boost`、`time_ps`、`phase`、`parameter_revision` 和
  `slot_to_identity`。方法频率与轨迹频率取并集，resume 按检查点裁剪。
  `output.report_gamd: false` 显式关闭此记录后不能做对应 CE2 分析。
- **`gamd_metadata.json`**：能量/时间单位、通道顺序及帧语义。
- **`gamd_calibration.json`**：每通道 Vmax/Vmin/σV 样本与选定的
  (threshold, k)——唯一参数来源。
- **resume**：`continue_md: true` 恢复已记录阶段、样本和参数，
  不重复已经完成的阶段；生产参数冻结。旧 total/dual 路径仍支持恢复。
- **reweighting**：GaMD 使用专用 CE2 分析入口，不读取 MetaD hills。
  见 [GaMD analysis](gamd-analysis.md)。

## 显式相互作用选择

```yaml
gamd:
  energy_unit: kcal/mol
  calibration_steps: 10000
  calibration_interval: 100
  preparation_steps: 1000
  boost_preparation_steps: 1000
  equilibration_steps: 10000
  adaptation_interval: 100
  frequency: 200
  boosts:
    - label: selected
      sigma0: 9.0
      threshold_policy: upper
      selectors:
        - atom_indices: [10, 11, 12]
          force_types: [bond, angle, dihedral, nonbonded]
    - label: remaining
      remaining: true
      sigma0: 6.0
      threshold_policy: lower
```

索引是准备后 System 的零基粒子编号，不是 PDB serial。列出的任意原子
参与相应类型的相互作用即选中**完整项**，全部参与原子都得到该项的力。
多个 selector 取去重并集；多个 boost 冲突时，列表中最早定义者拥有该项。
`remaining` 可省略，最多一个且必须最后出现；省略时残余项保持原强度。
用户 restraint 永远不进入此选择，也不会进入 remainder。

每通道独立选择 sigma0 和 `lower`/`upper` 阈值；upper 不合法时回落 lower。
设置可明确使用 kcal/mol（乘 4.184 转换），输出统一 kJ/mol。
顺序进入 fingerprint 和恢复兼容检查；不能在 resume 时交换 boost 顺序。
分解报告保留匹配、已分配、已被前序认领及未 boost 的信息。

非键分解保留原 NonbondedForce 的 PME、例外、switching 和色散修正，
通过剩余原子集合的完整能量差定义通道。它需要多份非键求值，开销随
通道数增长；最多 32 个后端能量组件，超过容量或不支持的基础 Force
表示会在创建 Context 前拒绝。不要用 cutoff 近似替代失败的 PME 分解。

阶段依次为 cMD 准备、cMD 统计、boost 准备、自适应平衡、固定生产。
除统计外的可选阶段可设为 0；`steps` 始终是整个运行的最终绝对步数。
生产分析排除准备和自适应阶段。统计退化时对应通道明确停用。
配置的检查点频率覆盖所有阶段，写入前先保存统计量和已完成的交换。
在线修改的 `sigma0` 和控制修订号随内核检查点原子保存；恢复时不会采用
检查点之后的修改，即使两次操作发生在同一积分步。

上面的 any-atom 规则会包含跨口袋边界的 bonded 项，因此不能自动声称
复现 LiGaMD2 论文的 pocket-internal bonded 分区。完整可见索引的模板和
可运行 peptide smoke 位于
[examples/ligamd2](https://github.com/NeoBinder/NeoDynamics/tree/main/examples/ligamd2)。
短软件检查不证明 binding PMF 收敛或结合动力学。

## 可选多配体交换

`gamd.exchange` 显式列出 `copies`（每份配体按等价原子顺序排列的索引）、
`designated_slot`、`reference_atoms`、`ligand_reference_atoms`（每份 copy
内部的偏移索引）、`interval`（默认每个积分步）、可选 `cutoff_nm` 及 `periodic`。
候选集必须满足化学、质量、力场参数和约束等价。配置 `cutoff_nm` 时，
只要当前 boosted 配体的距离 `<= cutoff_nm` 就保持不动，无论其他配体是否更近。
仅当当前配体距离 `> cutoff_nm`，且其他配体中有距离 `<= cutoff_nm` 的候选时，
立即交换其中最近者；平局按输入顺序，没有能量或 Metropolis 接受判定。
未设置 cutoff 时保留所有 slot 中取最近者的
选择。显式 `interval` 可调整检查频率；默认 1。交换整体置换位置和速度，
先于报告与检查点，身份
由事件账本追踪；分析必须声明跟踪固定 slot 还是物理 identity。
交换检查也覆盖 cMD 准备、统计、boost 准备和自适应平衡阶段。
`ligand_exchange.json` 保存 schema 2 元数据及当前事务状态，
`ligand_exchange_events.jsonl` 仅追加真正发生的交换；未交换的检查不写日志。
恢复与分析需要同时保留这两个文件。包含虚拟位点或作为虚拟位点父原子的
配体映射会在运行前拒绝。

此确定性最近配体交换是实验性的 earlier-LiGaMD 扩展。它可以改变 boost
能量；保持原始物理能量和动能并不足以建立有偏平衡分布。
默认 CE2 拒绝将此类运行解释成平衡 PMF，显式诊断覆盖也不会建立该声明。

## 参考文献

- Miao, Feher & McCammon, *J. Chem. Theory Comput.* 11, 5208（2015）——
  [GaMD 原始论文](https://doi.org/10.1021/acs.jctc.5b00436)。
- Miao, *J. Chem. Phys.* 152, 244108（2020）——
  [LiGaMD](https://doi.org/10.1063/5.0005907)。
- Copeland, Miao 等, *J. Phys. Chem. B* 126, 481（2022）——
  [gamd-openmm](https://doi.org/10.1021/acs.jpcb.1c0864)（其 Langevin
  更新形式被移植；乘性双 boost 缩放被否决，见 ADR-0005）。
- Wang & Miao, [LiGaMD2](https://doi.org/10.1021/acs.jctc.2c01194)。
- Miao, Bhattarai & Wang, [LiGaMD](https://doi.org/10.1021/acs.jctc.0c00395)。
- [ADR-0005](../adr/0005-gamd-boost-seam.md) 与
  [ADR-0009](../adr/0009-ordered-potential-boosts.md) —— 内核与选择契约。
