# GaMD（method: gamd）

> 需求与设计决策：[issue #10](https://github.com/NeoBinder/NeoDynamics/issues/10) ·
> [ADR-0005 GaMD boost 内核缝](../adr/0005-gamd-boost-seam.md)（BoostOps seam、
> 加性多通道缩放、全部被否决的替代方案）·
> [ADR-0012 分阶段步数与重跑](../adr/0012-gamd-staged-steps-and-restarts.md)
> （五阶段 steps 映射、边界检查点、restart_from 语义、参数推送的 AMBER 调度：
> 阶段2 段末一次、阶段4 内按 ntave 窗口）与
> [ADR-0013 阶段即独立 leg](../adr/0013-gamd-stage-legs.md)
> （每阶段独立 Context、步数从 0 起）

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

NPT 续跑：GaMD 与普通 MD/MetaD 一样默认使用 OpenMM 原生体积提议——安装 boost
不会接管它（boost 是 System 里的真实力，原生接受判据本已包含 boost 势能）。
OpenMM 检查点会恢复 barostat 的随机流（上游 7.4.1 起），但**不保存**提议宽度、
接受计数与更新相位，因此同一段中途续跑得到的是**统计等价**而非逐步位一致的
轨迹；`gamd.restart_from` 从阶段边界重入，不受此影响。只有显式
`barostat: {pressure: 1.0, frequency: 25, managed: true}` 才启用适配器拥有的
提议状态（检查点里额外的 `NEOMD-NPT-1` 块）以换取 barostat proposal 的逐步恢复；但所有 GaMD 的 CustomIntegrator 不
序列化自身 RNG 流，所以即使启用 managed，中途 resume 也只是统计等价，
而不是逐步位一致。managed 模式改变随机数序列，不能与原生 NPT 轨迹逐步
比较，也不能混用两种检查点。通用的续跑
方案（固定网格回收 Context）尚未实现，见
[ADR 0009](../adr/0009-ordered-potential-boosts.md)。

## 使用

GaMD 换入 `method: gamd`、一个按阶段的 `steps:` 映射和一个 `gamd:` 段。
通道布局只有一种词汇表：**`boosts`**（可选的有序 boost 声明列表）——
每条 boost 用选择器（原子组 + 力类型 + 跨界策略）独立圈定自己的能量项，
**声明之间允许重叠**（AMBER dual = 一条全体系 boost + 一条全体二面角 boost，
共享的二面角被两个通道各自加速一次）；可选的最后一条
`remaining: true` boost 精确取其余所有声明覆盖并集的补集。
不写 `boosts` 时，plan 归一化会补出一条显式的默认 `boost1`：`atom_indices: all` + 四类力（bond/angle/dihedral/nonbonded）、`threshold_policy: lower`、`sigma0: 6 kcal/mol`（AMBER igamd=1 的全体系 total boost）。它走与显式 boost 完全相同的分区/标定流程，标签是 `boost1`；因此不写 `boosts` 与手写这条 boost 等价。
（旧的 `mode`/`dual` 与力组 ID 式 `channels` 拼写已删除：报错并给出迁移提示。）
`sigma0`
为 boost 强度旋钮：**不写时默认 6 kcal/mol（=25.104 kJ/mol）**，写数值时单位默认 kJ/mol（段级 `energy_unit` 可改成 kcal/mol）。
一个 yaml 覆盖完整协议——
五个阶段依次是 `unbiased_eq`（无偏平衡1，cMD prep）、`unbiased_production`
（无偏平衡2，即标定统计，AMBER 的 cMD 统计段）、`biased_eq`
（boost 平衡1，AMBER 的 ntebprep：冻结运行）、`biased_eq2`
（boost 平衡2，可选，AMBER 的 nteb：**段内自适应**）、`production`
（固定生产），速度在阶段间连续携带；boost 安装时零强度。

**boost 参数的推送遵循 AMBER 的调度**（用户确认；ADR-0010 修订）：

```
unbiased_production 段末 ──►  推送一次（这套参数启动所有 boost 段）
biased_eq             段内 ──►  冻结：不更新统计、不推送
biased_eq2            段内 ──►  每关闭一个 ntave 窗口推送一次，段末再推一次
production            段内 ──►  冻结 biased_eq2 结束时的最后一套
```

统计配方与 AMBER/GaMD-OpenMM 完全一致：**统计段（阶段2/4）的每一步**都
读取被 boost 区域的 V0 并更新计算值——Vmax/Vmin 自统计开始连续累计（从
cMD 段带进自适应段），窗口 Welford（Vavg/σV）逐步累加，`ntave` 步的窗口
**在边界步上精确关闭**（GaMD-OpenMM 的 windowCount 语义），样本不足的窗
口回退到全段累计统计。`steps.ntave` 不写时按 AMBER 建议取体系总原子数的
4 倍吸附到 1-2-5 系列（`default_ntave`；如 35633 原子 → 100000）。**与
AMBER 对齐的整窗约束**：`steps.unbiased_production` 与 `steps.biased_eq2`
必须是 `steps.ntave` 的整数倍（AMBER 原文 "ntcmdprep, ntcmd, ntebprep and
nteb need to be multiples of ntave"），保证每次推送读到的窗口恰好完整。
V0 的读取走 AMBER 的语义：**整力选择的通道**（如 `atom_indices: all` +
`force_types: [dihedral]`，或唯一的 `remaining: true`）会把被选中的原生力
搬进该通道自己的纯净 force group（组内没有别的力），V0 = 该 group 的能
量；**部分选择**的通道（子集原子/跨界项）没有对应的纯 group，V0 从
CustomCVForce 的 u 克隆值读（两条路径数值恒等，`verify_cv_v0` 交叉验
证）。读取机制现在不再按平台或 V0 来源分叉：**所有 GaMD 都使用一个
CustomIntegrator**（LFMiddle 算子与 stock Langevin 逐位等价，无噪声下
机器精度；噪声抽取顺序不同，轨迹确定但与 stock 不逐位）。group-readable
通道的尾部 op 把每步 V0 写进积分器全局变量：非 debug 为
`gamd_p_<label>`，debug 为 `out9_<label>`；CV/clone 通道没有 fast global，
由 `boost_readings()` 在当前帧读 `_base_group_energies()`。逐点统计优先
读 group fast global（`boost_step_v0()`），缺失的 channel 按 label 补
fresh 读取。计算不再把 stock 作为 CPU 默认，也不再用
`NEOMD_GAMD_STATS_INTEGRATOR` 选择积分器。
E/k 只是推送时刻从这些统计现算的推导值。每次推送记一个单调 `revision`
（gamd.tsv 每行记录其当时的 revision），参数写入 `gamd_calibration.json`
并经 `set_boost_param` 推送到内核。逐点统计状态和当前参数随检查点信封
（`gamd_stats`，与 `gamd_leg` 同一个 NEOMD 包装）原子持久化，所以自适应段
中途崩溃的续跑会从恢复点**精确恢复统计状态**（含段内窗口推送）；但所有
GaMD 现在都使用 `CustomIntegrator`，其检查点不序列化 RNG 流，所以后续
随机轨迹只是**统计等价**，不是逐位一致。`output.gamd.sample_interval`（默认 1000）只控制 journal 的**诊断**样
本密度（`gamd_stages.json` 的 samples），不再是统计来源；**五个阶段全部必
填、全部 ≥ 1 步**（严格五阶段协议；唯一例外：外部 `calibration_input` 供
参数时 `unbiased_production` 可为 0——它的统计段无事可做）。boost 平衡段
的统计太平（< 2 个样本）或没有可用方差时，会保留当前安装的参数而不是把
通道关掉。
最小可运行 plan（默认 boost1）；restraint 墙只是给标定一点势能起伏：

```yaml
method: gamd
steps:                     # 每阶段所需步数（不考虑已跑步数）
  unbiased_eq: 50000       # 无偏平衡1（cMD prep）
  unbiased_production: 50000  # 无偏平衡2 —— 在此采样统计，段末推送 E/k
  biased_eq: 10000         # boost 平衡1（ntebprep）——用无偏标定的参数，冻结
  biased_eq2: 50000        # boost 平衡2（nteb）——每 ntave 窗口自适应
  ntave: 50000             # Vavg/σV 窗口步数（不写则按 ~4x原子数 吸附 1-2-5）
  production: 340000       # 冻结 biased_eq2 结束时的最后一套参数
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
  progress_interval: 5000   # 控制台进度行（阶段/百分比/速度/预计结束）的步长（默认 5000）
  gamd:                    # GaMD 自己的输出细节都放在这个命名空间里
    report: true           # 写 gamd.tsv（默认开）
    sample_interval: 50    # journal 诊断样本的步长（默认 1000；统计本身每步更新）
    trace_interval: 10     # production 期间写 gamd.tsv 的步长（默认与
                           # trajectory_interval 同步——每个存帧一行；无轨迹时 5000）
restraint:                 # 可选，建议标定前加
  dist: {type: distance, grp1: "0", grp2: "1", restr_k: 500.0, max_nm: 0.8}
gamd:
  sigma0: 6.0              # 写了就是 6 kJ/mol（默认单位）；不写则 6 kcal/mol
  # 不写 boosts = 默认 boost1（全体系四类力，sigma0=6 kcal/mol，lower）。
  # 写 boosts 则每条独立圈定、允许重叠；最后的 remaining: true 取并集补集：
  # boost_cv_mode: dv | v0_dv（默认 dv）
  #   dv    = V0 保持原生力、CustomCVForce 只承载 dV（默认，统计快路径可用）
  #   v0_dv = 被完整覆盖的原生力被移除，CV 承载该 V0 + dV（少一次非键求值；
  #           要求所有被 boost 的 Force 都被通道完整覆盖，否则报错提示改 dv）
  # emit_system:             # 写了就不跑模拟：只生成 <stage>_system.xml
  #   stage: production      # 默认 production
  #   E: {boost1: -1000.0}   # 每通道 threshold，kJ/mol（缺省 0）
  #   K: {boost1: 0.001}     # 每通道 k，1/(kJ/mol)（缺省 0）
  # boosts:
  #   - label: total      # AMBER igamd=3 dual 的前半：全体系
  #     selectors: [{atom_indices: all, force_types: all}]
  #   - label: dihedral   # 后半：全体二面角（与前一条共享二面角项）
  #     selectors: [{atom_indices: all, force_types: [dihedral]}]
```

**每个阶段就是一个 leg**：ADR-0011 给每个阶段单独的 Context 和单独的
步数轴，`steps.<阶段>` 同时是该阶段的协议长度**和该 leg 的长度**，阶段内
步数、`time_ps` 都从 0 起；跨阶段没有全局绝对步数（`steps` 的派生总和
只是编译期便利值）。每跨一个阶段边界都会用派生种子
（`seed + leg 序号`）重建 Context，坐标/速度/盒子原样交接，Langevin 噪声与
体积提议状态重新开始，因此每个 leg 的随机流独立且可复现。leg 边界因而也是
唯一有保证的重入点（中途续跑在原生 NPT 下只是统计等价，见上）。非 gamd 方法的 `steps` 仍是单个
整数。GaMD 不与 metadynamics 组合（安装 boost 后再 `install_bias`
直接报错）。

## 产物

GaMD 是**分阶段**方法，因此随时间更新的记录**每个阶段各存一份**，文件名用
阶段名做前缀（其它方法仍沿用 `output.state` / `output.dcd` / `restraint.tsv`）：

| 每阶段一份 | 说明 |
|---|---|
| `<阶段>.state` | 该阶段的热力学行（openmm StateDataReporter 兼容格式） |
| `<阶段>.dcd` | 该阶段的轨迹帧（每段一个 DCD，各自带表头/首步/步长） |
| `<阶段>_restraint.tsv` | 该阶段的 restraint 观测量 + 偏置能量 |
| `<阶段>.ckpt` | 该阶段的滚动检查点；跟随 `output.checkpoint_interval`，与全局 `output.ckpt` **同时更新** |

阶段名为 `unbiased_eq` / `unbiased_production` / `biased_eq` / `biased_eq2` /
`production`；**步数为 0 的阶段不跑，因此不产生对应文件**。全局文件与它们并存：

- **`output.ckpt`**：唯一的滚动恢复点（**阶段中续跑** `continue_md` 的入口，
  由 `input_files.checkpoint` 显式指定；阶段头续跑则用 `restart/<阶段>.ckpt`），
  每个检查点都写。所有检查点（`output.ckpt`、`<阶段>.ckpt`、`restart/<阶段>.ckpt`）
  的元数据里都记着该状态属于哪个 leg（键 `gamd_leg`）：leg 的步数与随机流都是
  Context 私有的，journal 又可能领先或落后于检查点，所以 leg 必须随状态一起保存，
  否则"步数 0"无法区分"本 leg 起点"和"下一个 leg 起点"。
- **`restart/<阶段>.ckpt`**：阶段**起点**检查点（`restart_from` 用）：
  `restart/<阶段>.ckpt` 是 `<阶段>` 这条 leg 的起点状态（在该 leg 的
  `start_leg` 之后立刻写入，即该 leg 的步 0），文件名就是它开始的阶段；
  零长度阶段与其后继共用同一状态。短于检查点间隔的阶段只有这个，
  没有顶层 `<阶段>.ckpt`。
- **`last.ckpt` / `last.pdbx`**：只在 production 跑完后写一次的收尾产物，不分阶段。

- **`gamd.tsv`**：保存构型处的每通道 ΔV / 原始能量 P / 力缩放 s，
  以及 `total_boost`、`time_ps`、`phase`、`parameter_revision` 和
  `slot_to_identity`。`output.gamd.trace_interval`（默认与 `trajectory_interval` 同步，无轨迹时 5000）与
  `output.trajectory_interval` 取并集：**每一帧都保证有对应行**（CE2 按步精确配对的硬要求），
  没有轨迹输出时，它就是这个磁带的唯一栅格。resume 按检查点裁剪。
  **只在 production 写**（准备阶段的样本在 `gamd_stages.json.samples`）；
  `output.gamd.report: false` 显式关闭此记录后不能做对应 CE2 分析。
- **`gamd_metadata.json`**：能量/时间单位、通道顺序及帧语义。`gamd_stages.json` 没有内嵌单位：`samples.energies`、`parameter_history` 的 threshold/sigma/vavg/vmax/vmin 都是 kJ/mol，`k` 是 1/(kJ/mol)，step 是 leg 内步数。
- **`gamd_calibration.json`**：每通道 Vmax/Vmin/σV 样本与选定的
  (threshold, k)；`calibration_input` 续跑时附带来源路径与 sha256。
- **`gamd_stages.json`**：阶段账本（schema 4）——模式、每阶段
  configured/completed（**每个阶段在自己的 leg 内计数**）、当前 leg 名
  `phase` 与该 leg 内的 `step`。两个核心段：
  - `samples`：**按阶段分组** `{<阶段>: [{step, energies:{<通道>: P}}, …]}`；
    只有 `unbiased_production` / `biased_eq` / `biased_eq2` 会采样（`biased_eq`
    的样本只做诊断，不参与参数统计），其余阶段是空表
    （它们职责不同，不共用一张总表）。
  - `parameter_history`：**按阶段名索引** `{<阶段>: {revision, step, channels}}`，
    只出现在真正推送过参数的段（`unbiased_production`——段末一次；
    `biased_eq2`——每个 ntave 窗口一次、段末一次，条目被后一次推送覆盖，
    `revision` 单调递增；外部输入记为 `external`）。`revision` 就是
    `gamd.tsv` 里 `parameter_revision` 列的值；`channels` 含
    vmin/vmax/vavg/sigma/threshold/k/bound 等。
  同目录续跑的唯一 journal（没有全局 `stage_ends`/`start_step`/
  `final_step` 键）；`gamd_calibration.json` 只是派生展示，删除它不妨碍续跑。
- **`gamd_partitions.json`**：分区报告（用户/调试产物）：`cv_mode`、每通道
  `groups`（**只列真实物理 force group**；clone-only lane 为 `[]`，不占
  force-group id）、`physical_force_group`（真实组或 `null`）、
  `matched/assigned/claimed/force_types/v0_source`、`residual`（只保留
  `assigned` 计数，没有组号）、`component_count`、`base_terms`、
  `removed_force_indices`、粒子与键的索引映射。
  它不在续跑路径上读取（续跑布局校验读 `gamd_stages.json` 的
  `fingerprint_payload`），可整份删除而不影响恢复。
- **`restart/<阶段>.ckpt`**：每个阶段 leg 的起点检查点，文件名即该 leg
  开始的阶段（零长度阶段与其后继写入同一状态）。滚动 `output.ckpt` 只保存
  最终状态，阶段重跑必须走这些 restart 检查点。
- **续跑（`continue_md: true`）**分两种场景：
  - **阶段中续跑**（不写 `gamd.restart_from`）：**必须**显式给
    `input_files.checkpoint`（gamd 目录没有隐式默认）；从该检查点恢复，
    journal 记下它属于哪个 leg，leg 内进度取恢复后的内核时钟，只补该 leg
    的剩余步数，续跑轨迹逐步位等；阶段步数可以随时改。若该检查点指向
    非 gamd 检查点（如普通平衡产物）则为**采纳**：采纳坐标与速度、
    **丢弃其步数轴**（leg 没有全局轴），五个阶段都从 0 起跑。
  - **阶段头续跑**（写 `gamd.restart_from: <阶段>`）：**无需**指定
    `input_files.checkpoint`——plan 自动把 `restart/<阶段>.ckpt`（该阶段
    起点）作为 continue_md 的入口；若显式指定，必须指向该文件或它的
    逐字节副本，否则报错。该 leg 从步 0 重新进入，阶段参数钉在边界
    标定值，每份分阶段磁带按**自己 leg 的长度**裁剪（resume 仍是裁剪的
    唯一 owner）。若续跑时改动了
  boost 平衡段的布局（增删 `biased_eq2`），链条末尾随之改变：新加的
  `biased_eq2` 自己跑自己的自适应（重放恢复的统计后按窗口推送）；删掉它
  时 prepare 会按 `unbiased_production` 恢复的统计重新推导并推送
  production 的参数，避免继承一个已不存在的阶段布局。边界缺失（前一阶段
  从未完成）会明确报错。
  biased 阶段重跑若无边界标定，需要 `gamd.calibration_input` 指向
  源运行的 `gamd_calibration.json`（逐字采纳）或能量磁带
  （`gamd.tsv` 的 `<label>__energy` 列 / `state.tsv` 的 potential
  列，现场重算标定）。**续跑/重跑允许你在 YAML 里改大部分参数**：
  `sigma0`、`threshold_policy`、`energy_unit`、每个 boost 的强度、采样/磁带步长、
  `exchange`、阶段步数、`restart_from`、`calibration_input` 都可以改——它们只影响
  "从检查点往后"的行为（恢复出的 Context 仍用它那一段开始时的参数，已经结束的阶段
  保留自己的统计），改动会以一条**点名漂移键的 WARNING** 记下来，不会静默生效。
  唯一不能改的是 **boost 布局**（有序列/list 里的通道顺序、每个通道的
  `atom_indices`/`force_types`/`include_boundary`、谁是 `remaining` 补集）：
  恢复出的状态属于那个 System 的力结构，journal 里钉住的参数也按通道名索引，所以
  布局变了会明确拒绝并给出"记录值 → 当前值"；要换布局就用 `input_files.checkpoint`
  把 `output.ckpt` 采纳成新 run。旧目录（journal 里没有这份描述）按其
  `gamd_metadata.json` 记录的配置判断。旧版 `gamd_stages.json`（schema 1）
  与旧 schema 目录（1/2）不再可直接续跑：用 `input_files.checkpoint`
  采纳其 `output.ckpt` 开新 run。
- **reweighting**：GaMD 使用专用 CE2 分析入口，不读取 MetaD hills。
  见 [GaMD analysis](gamd-analysis.md)。

## 显式相互作用选择

```yaml
steps:
  unbiased_eq: 1000
  unbiased_production: 10000
  biased_eq: 1000
  biased_eq2: 10000
  production: 100000
output:
  gamd: {sample_interval: 100, trace_interval: 200}  # 平衡段统计 / production 磁带
gamd:
  energy_unit: kcal/mol
  boosts:
    - label: pocket
      sigma0: 9.0
      threshold_policy: upper
      selectors:
        - atom_indices: [10, 11, 12]   # 底物 + 口袋 residue
          force_types: all             # 等价于 bond, angle, dihedral, nonbonded
          # include_boundary: {bond: true, angle: false}   # 可选，按力类型覆盖默认
    - label: remaining
      remaining: true
      sigma0: 6.0
      threshold_policy: lower
```

索引是准备后 System 的零基粒子编号，不是 PDB serial；也可写标量 `all`
（**全体系**，AMBER `igamd=1..5` 的语义——例如 `atom_indices: all` +
`force_types: [dihedral]` 就是"对全部二面角加 boost"，等价于 AMBER 的
`igamd=2`）。`force_types` 可写 `bond`/`angle`/`dihedral`/`nonbonded`
的列表，或标量 `all`（四类全选）。
每个被选中的力类型有一个布尔开关 `include_boundary`：

- `false`：只选中**全部原子都在组内**的项（跨组项不入本 boost）；
- `true`：任一原子在组内即选中，包含跨组项。

默认值：`bond`/`angle`/`dihedral` 为 `false`，`nonbonded` 为 `true`。
`include_boundary` 里只能写该 selector 的 `force_types` 已经选中的力类型，
写了别的类型直接报错（避免开关静默失效）。

因此默认的"底物 + 口袋"selector 拿到组内键合项、组内非键对，以及所有至少
一个原子在组内的非键对（组↔蛋白其余、组↔环境的远程力）；跨口袋边界的
bond/angle/dihedral 默认**不**入该 boost，落到后面的 `remaining`（没有
`remaining` 时保持未 boost、正常跑）。要改变跨界行为就显式覆盖，例如
`include_boundary: {bond: true, angle: false}` 让跨界 bond 入、跨界 angle
不入；`include_boundary` 也可写成标量 `true`/`false`，作用于该 selector
列出的全部力类型。同一 boost 的 nonbonded selector 必须使用同一个值。

`include_boundary: {nonbonded: false}` 表示组自身的孤立子体系非键能量
`M(S)`：组内直接项与例外、倒空间的 `Q(S,S)` 块、自能、按组净电荷的中和
背景、按组原子的色散修正。PME 倒空间是所有电荷的全局函数，`S|C` 交叉项
没有原子对归属，所以这不是"逐原子对都在组内"的字面过滤，而是代数块定义；
补集 `M(all) − M(S)` 由 `remaining` 承接，两者相加精确等于完整非键能量。
掩码子集能量只在原子集合互不相交时可加，因此 `nonbonded: false` 的集合不能
与同一分区的其他非键认领共享原子——校验层会直接拒绝。

多个 selector 取去重并集；多个 boost 冲突时，列表中最早定义者拥有该项。
`remaining` 可省略，最多一个且必须最后出现；它永远是尚未被认领的精确补集，
不需要写 `include_boundary`。两个典型形态：

- 只选 nonbonded（`force_types: [nonbonded]`，`nonbonded` 默认 `true`）：
  `remaining` = 全部 bond/angle/dihedral + **两个原子都在组外**的非键对；
- 四类全选且 `bond`/`angle`/`dihedral` 为 `false`、`nonbonded` 为 `true`：
  `remaining` = 跨原子组的 bond/angle/dihedral + 剩余原子组内部的全部力
  （含组内非键）。

上面两种 `nonbonded: true` 的情形里，`remaining` 的非键就是一个普通的
"组外原子掩码子集"。只有当某个 boost 用 `include_boundary: {nonbonded:
false}` 拿走"组内非键块"时，补集才表现为差表达式 `M(all) − M(S)`。
用户 restraint 永远不进入此选择，也不会进入 remainder。

每通道独立选择 sigma0 和 `lower`/`upper` 阈值；upper 不合法时回落 lower。
设置可明确使用 kcal/mol（乘 4.184 转换），输出统一 kJ/mol。
顺序进入"boost 布局"检查：**续跑不能交换 boost 顺序或改动某个通道的选择**（但可以改
各通道的 sigma0/threshold_policy/energy_unit）。
分解报告保留匹配、已分配、已被前序认领及未 boost 的信息。

非键分解保留原 NonbondedForce 的 PME、例外、switching 和色散修正，
通过剩余原子集合的完整能量差定义通道。它需要多份非键求值，开销随
通道数增长；最多 32 个后端能量组件，超过容量或不支持的基础 Force
表示会在创建 Context 前拒绝。不要用 cutoff 近似替代失败的 PME 分解。

阶段依次为无偏平衡1（cMD 准备）、无偏平衡2（标定统计）、boost 平衡1、
boost 平衡2（可选）、固定生产；`gamd.tsv` 的阶段列使用同一组名字
（`unbiased_eq` / `unbiased_production` / `biased_eq` / `biased_eq2` /
`production`）。五个阶段全部必填且各 ≥ 1 步（严格五阶段协议）；
每个阶段各自是一个 leg（步数从 0 起，跨阶段没有绝对步数，`steps` 的派生
总和只是编译期值）。生产分析排除准备阶段。统计退化时对应通道
明确停用（boost 平衡段的样本太少时改为保留上一段交来的参数）。
配置的检查点频率覆盖所有阶段，写入前先保存统计量和已完成的交换。
`sigma0` 是**计划输入**：不写时缺省 6 kcal/mol（=25.104 kJ/mol），写数值时单位默认
kJ/mol（段级 `energy_unit` 可覆盖）。它在整条运行中**恒定不变**，也不作为运行状态
或记录项出现在任何产物里（journal、检查点元数据、标定文件里都没有它）；阶段边界
重算的是由它派生的 (E, k)。要改 boost 强度就改 YAML 并从相应阶段重跑，没有运行中
调参的入口。

默认的 bonded 开关就是 LiGaMD2 的 pocket-internal 分区；写成
`include_boundary: {bond: true, angle: true, dihedral: true}` 则回到早期
any-member 约定。两种写法都不构成对论文数值的精确复现声明。完整可见索引的
模板和可运行 peptide smoke 位于
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
交换检查也覆盖无偏平衡、统计与两个 boost 平衡阶段。
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
