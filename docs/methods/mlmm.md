# ML/MM 耦合（ML-potential region）

- issue：[#12](https://github.com/cyrushu/NeoDynamics/issues/12)（QM/MM 修复与完整化 → ML/MM 切片）
- 实现状态：W2-d 已落地（ligand-only ML 区；活性位点残基跨界处理为 W3-c）
- 决策记录：[ADR-0004](../adr/0004-mlmm-in-tree-coupling.md)

## 背景与动机

issue #12 的原始诉求是修复 v1 `src/neomd/qmmm/pipeline.py`：其中
`prepare_qmmm()` 存在多处未定义引用（`topology_export`、`OpenMMWrapper`、
`QMMM_Subtractive_Handler`、`qm_wrapper`、`qmmm_force`），README 却把
QM/MM 列为待发布功能。**判定：legacy qmmm 是未完成重构的 WIP，不是
bug，不在 `neomd_legacy` 的 bug-fix-only 范围内**——v1 硬冻结（固定决策
#1）下不修 v1 代码，新功能落 v2。

issue 技术方案的第二步给了两条路：A. ML/MM 先行（NNP/MM 混合，成本远
低于 DFT QM/MM，2024–2026 文献已验证 ML/MM 在酶反应位点的可用性）；B.
真 QM/MM（ORCA 外部驱动 + link atom + 电荷位移）。仓库已有 QM 工具链
基础（`bin/resp2_orca.py` 集成 ORCA + Multiwfn），酶活性位点（如
P450 Cpd I）场景对 QM/MM 有真实需求。

**2026-09-02 范围决定**（issue 开发计划卷首研究决定 + W2-d 行）：真
QM/MM（ORCA 后端、link atom、电荷位移）**暂缓**；ML/MM **转入正式
开发**，且**不走 openmm-ml 依赖**，自研轻量耦合模块 + 自有模型加载器。
真 QM/MM 留待重启（另立 ADR）。

## 与 issue 方案的差异（v2 决策）

issue 设想"基于 openmm-ml（ANI-2x/MACE 等）实现 additive ML/MM"。
v2 落地时改为：

- **不依赖 openmm-ml**。评估结论（研究决定，2026-09-02）：openmm-ml
  通用层仅约 1.5k 行，其余约 65 KB 是 9 个逐模型适配器（aimnet2/ani/
  mace/nequip/deepmd/orb/torchmdnet/fennix/ase）——本仓库要接**自有**
  ML potential，这层整体用不上；且其活跃 churn 恰集中在模型注册表层。
  openmm-ml 降级为可选交叉验证参照（装了才跑的 marker-gated 对拍测试）。
- **`KernelSpec.ml_region` 内核规格字段（in-tree），不是插件**：与
  barostat / `dummy_exceptions` 同类——"Context 创建前对 System 的装配
  说明"，走 `Plan → run.build_kernel_spec → KernelSpec.ml_region →
  openmm 适配器` 的既有通道；fake 内核忽略之（文档化）。
- **机械嵌入逐字移植**（固定决策 #2 纪律）：openmm-ml
  `embeddings/mechanicalembedding.py` + `utilities.py` 的
  `removeBonds` / `addCustomNonbondedExclusions`，MIT，带 attribution
  与源 commit；上游 `makeCustomNonbondedExclusions` 笔误按定义名修正
  并注明。
- **通用 TorchScript 加载器**（openmm-torch TorchForce）：模型文件即
  接口，自有/公开 TorchScript 势一等公民，不要逐模型注册表。
- **两适配器纪律**（固定决策 #7 的 ML 版）：生产 = openmm 适配器 +
  TorchForce + TorchScript 模型；CI 默认门 = 同一 openmm 适配器 +
  **mock NNP**（标准 openmm custom force 拼装的确定性玩具势，无 torch
  装机即可走通全管线）；fake 忽略。

## 使用

README 的「ML/MM coupling」一节是入口摘要；本节为完整版。

一个 plan 段落把一个区域（本期 ligand-only）变成 ML 势区域：

```yaml
ml_region:
  indices: [1234, 1235, 1236]        # 0 基粒子索引（或 "1234,1235,..."）
  model:
    type: torchscript                # 或: mock
    path: my_nnp.pt                  # torchscript：模型文件即接口
    long_range_electrostatics: false # 周期体系必须声明
    periodic: true                   # 可选；默认随体系
    # mock 专属参数：tether_k (500 kJ/mol/nm^2)、repulsion_k (1 kJ/mol)、
    #               repulsion_sigma (0.15 nm)
```

- 校验走 collect-all（yaml key path + did-you-mean）；`neomd validate
  plan.yaml --check-files` 额外检查 indices 界内与 path 存在性。
- 端到端示例：[examples/mlmm_ligand](../../examples/mlmm_ligand)
  （3HTB + JZ4 配体区，min + 100 ps）。
- torch 层测试：`pixi run -e ml test-ml`（默认门保持 torch-free）。

## 架构与产物

- **装配位置：openmm 适配器内、Context 创建前。** NNP Force
  （TorchForce）不可 XML 序列化，因此绝不在 prepare 层把 ml_region 写
  进 `system.xml`；`OpenMMKernel.__init__` 反序列化 System 后、惰性
  `simulation` 属性创建 Context 前，由 `neomd.ml.assemble` 完成机械
  嵌入 + NNP 力添加。力组 id 由唯一分配器 `port.pick_free_force_group`
  发放。
- **机械嵌入语义**（与 openmm-ml 一致）：ML 原子的 MM 点电荷不置零
  ——它们继续承担 ML↔MM 静电相互作用；被移除的是 ML-ML 的 MM 键合项
  （经 XML 往返删 Bonds/Angles/Torsions）与 ML-ML 非键相互作用
  （exception 置零）。此版本无电荷再分布。
- **两层模型**：
  - `torchscript`——TorchForce 加载 `.pt`。单位契约（必须精确）：模型
    收到**整个体系**的坐标（`float32`，`(N, 3)`，**nm**；TorchForce
    无原子子集参数，模型必须把 ml_region 索引烘焙进 forward 内部），
    周期体系另传 nm 盒向量 `(3, 3)`，返回标量能量 **kJ/mol**；Å/eV/
    kcal 训练的模型在 forward 内换算。
  - `mock`——tether + 软排斥的确定性玩具势（非物理），CI 层保障：
    无 torch 装机即可走通 spec 解析 → 机械嵌入 → 适配器装配 → 运行
    全管线。
- **`long_range_electrostatics`**：TorchScript 模型无法被探测，周期
  体系下该声明决定机械嵌入支路——`false`（默认）= ML-ML 非键经零值
  exception 直接移除；`true` = ML-ML 库仑按真实电荷乘积保留，由仅含
  ML 电荷的 PME 力经 CustomCVForce 减去。
- **环境解析**：conda-forge openmm-torch 1.5.1 全部构建的 openmm 上界
  `<8.6.0a0`，与仓库 pin 的 8.6 不可共解；`ml` pixi 环境临时 pin
  `openmm = "8.5.*"` + `openmm-torch = "1.5.*"` + `pytorch = "2.12.*"`
  （solve-group `ml`，与默认环境隔离；升级 pin 是显式事件，固定决策
  #10 纪律）。

## 参考文献与 ADR

- [ADR-0004：ML/MM 以内核规格字段在树内实装，不做插件；不依赖
  openmm-ml](../adr/0004-mlmm-in-tree-coupling.md)
- [issue 开发计划](../issue-dev-plan.md) 卷首「研究决定（openmm-ml
  评估）」与 W2-d 行
- openmm-ml（MIT，机械嵌入移植源，v1.7 / commit `501c3a0`）
- REANN ML/MM, *JCTC* 2025（<0.5 kcal/mol、80× 加速——issue #12 引用的
  ML/MM 酶位点可用性文献）
