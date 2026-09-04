# ADR-0004：ML/MM 以内核规格字段（`KernelSpec.ml_region`）在树内实装，不做插件；不依赖 openmm-ml

- 状态：已确认（2026-09-03，issue #12 ML/MM 部分）
- 决策者：项目维护者
- 关联：[issue #12](https://github.com/NeoBinder/NeoDynamics/issues/12)（ML/MM，含 openmm-ml 评估）；[AGENTS.md](https://github.com/NeoBinder/NeoDynamics/blob/main/AGENTS.md) 固定决策 #2（物理逐字移植）、#7（双适配器纪律）、#10（openmm 版本 pin 纪律）；ADR-0001

## 背景

issue #12 拆分后 ML/MM 转入正式开发（真 QM/ORCA 暂缓）。2026-09-02 研究评估了
openmm-ml（1.3→1.7 五版，最后推送 2026-08-29，commit `501c3a0`）：其通用层
（`mlpotential.py` + `embeddings/`）约 1.5k 行，其余约 65 KB 是 9 个逐模型适配器
（aimnet2/ani/mace/nequip/deepmd/orb/torchmdnet/fennix/ase）。本仓库要接的是
**自有** TorchScript ML 势——那 9 个适配器与逐模型注册表整体用不上，而其活跃
churn 恰集中在注册表层。真正绕不开的底座是 openmm-torch（TorchForce C++ 插件）
+ torch，自研也省不掉。

## 决策

**三件事：**

1. **`ml_region` 是内核规格字段（in-tree），不是插件。** 它与 barostat /
   `dummy_exceptions` 同类——"Context 创建前对 System 的装配说明"，走
   `Plan → run.build_kernel_spec → KernelSpec.ml_region → openmm 适配器` 的
   既有通道；fake 内核忽略之（文档化）。插件机制（plan-schema
   namespace）服务的是"用户可装的第三方扩展"，而 ML/MM 耦合是内核装配
   知识，不满足插件的判据。

2. **自研轻量耦合模块 `src/neomd/ml/`，openmm-ml 不进依赖。** 机械嵌入
   （mechanical embedding）自 openmm-ml **逐字移植**（固定决策 #2 的纪律）：
   `embeddings/mechanicalembedding.py`（非插值路径）+ `embeddings/utilities.py`
   的 `removeBonds` / `addCustomNonbondedExclusions`，MIT，文件头带 attribution
   与源 commit。模型侧是**通用 TorchScript 加载器**（openmm-torch
   `TorchForce`，模型文件即接口，无逐模型注册表）+ **mock NNP**（标准
   openmm custom force 拼装的确定性玩具势，无 torch 也能跑通全管线）。源码
   中 `mechanicalembedding.py` 调 `utilities.makeCustomNonbondedExclusions` 而
   `utilities.py` 定义的是 `addCustomNonbondedExclusions`（上游潜在笔误，遇
   CustomNonbondedForce 会 AttributeError）——移植按定义名修正并注明。
   openmm-ml 降级为**可选交叉验证参照**：不进 pixi，`importorskip`
   gated 对拍测试。

3. **装配位置：openmm 适配器内、Context 创建前。** NNP Force
   （TorchForce）不可 XML 序列化，因此**绝不**在 prepare 层把 ml_region 写进
   system.xml；`OpenMMKernel.__init__` 反序列化 System 后、惰性 `simulation`
   属性创建 Context 前，由 `neomd.ml.assemble` 完成嵌入 + NNP 力添加（复用
   适配器既有的 pre-Context 路径）。力组 id 由唯一分配器
   `port.pick_free_force_group` 发放（固定纪律）。

## `KernelSpec.ml_region` 形状

```yaml
ml_region:
  indices: [1234, 1235, ...]      # ML 区粒子（0 基）；首期 ligand-only，
                                  # 跨边界残基 = 后续残基 ML 区附录
  model:
    type: torchscript | mock
    path: model.pt                 # torchscript 必填；模型文件即接口
    long_range_electrostatics: false  # torchscript 可选；见下
    periodic: true|false           # 可选；默认随体系
    # mock 专属参数：tether_k / repulsion_k / repulsion_sigma
```

- **单位契约（torchscript，必须精确）**：TorchForce 把**整个体系**的粒子
  坐标以 **nm**（`float32`，shape `(N_system, 3)`）喂给模型 `forward`——
  TorchForce 无原子子集参数，模型必须**在 forward 内部自选原子**（把
  ml_region 索引烘焙进模型，如 `index_select`，openmm-ml 各模型包装同款
  做法）；周期体系再传 **nm** 盒向量 `(3, 3)`；模型必须返回标量能量
  **kJ/mol**。训练用 Å/eV/kcal 的模型必须在自己的 `forward` 里换算（乘
  10 转 Å、能量乘回 kJ/mol）。
- **`long_range_electrostatics`**：TorchScript 模型无法被探测，周期体系下该
  声明决定机械嵌入走哪条支路（移植源逻辑）：`false`（默认）= ML-ML 非键
  相互作用经零值 exception 直接移除；`true` = ML-ML 库仑按真实电荷乘积保留，
  由一个仅含 ML 电荷的 PME 力经 `CustomCVForce` 减去。
- 机械嵌入语义（与 openmm-ml 一致）：ML 原子的 MM **点电荷不置零**——它们
  继续承担 ML↔MM 静电相互作用；被移除的是 ML-ML 的 MM 键合项（经 XML 往返
  删 Bonds/Angles/Torsions）与 ML-ML 非键相互作用（exception 置零）。源码
  此版本无电荷再分布。
- 验证走 collect-all（yaml key path + did-you-mean），`neomd validate
  --check-files` 额外做 indices 界内与 path 存在性检查。

## 两适配器纪律（固定决策 #7 的 ML 版）

- 生产 = openmm 适配器 + openmm-torch TorchForce + 用户自有/公开 TorchScript 模型；
- CI（默认门）= 同一 openmm 适配器 + mock NNP——**无 torch 装机**即可走通
  spec 解析 → 机械嵌入 → 适配器装配 → 运行 全管线；
- fake 内核**忽略** `ml_region`（文档化选择，非走 mock 数值：fake 的角色是
  driver/probe/method 的确定性测试底座，其"物理"是零力自由粒子，在那里数值
  复刻 mock 势不产生任何守护价值）。

## 环境解析（2026-09-03 实测）

- **conda-forge**：openmm-torch 最新 1.5.1（上游 v1.5.1，2025-02），其全部
  linux-64 构建的 openmm 上界均 `<8.6.0a0`（含最新 build 8：openmm
  >=8.5.2,<8.6 + pytorch 2.12）——与仓库 pin 的 openmm `8.6.*` **不可共解**。
- **PyPI**：openmm-torch 从未发布（simple index 404）。
- **结论**：`ml` 环境临时 pin `openmm = "8.5.*"` + `openmm-torch = "1.5.*"` +
  `pytorch = "2.12.*"`（solve-group `ml`，与 default/test/dev 的 8.6 隔离；
  ml 门只跑 `tests/v2/test_mlmm.py`，不触 openmm_privates 的 8.6 门）。固定
  决策 #10 纪律照记：这是 pixi.toml 里的显式 pin；conda-forge 出 tracking
  8.6 的 openmm-torch 构建后，升级该 pin 是一次显式事件。
- 默认门（`pixi run test`）保持 **torch-free**：真实 torch 测试全部
  `pytest.importorskip("torch")` / `importorskip("openmmtorch")` /
  `importorskip("openmmml")` gated，ml 环境用 `pixi run -e ml test-ml` 跑。

## 否决的替代方案

### 依赖 openmm-ml（MLPotential.createMixedSystem 全套）

- 65 KB 里 ~95% 是我们不需要的逐模型适配器；其注册表层 churn 最快；
- 模型注册表与"自有 TorchScript 模型一等公民"目标相反；
- 机械嵌入核心 <300 行，逐字移植 + attribution 后可控可测（交叉验证测试
  对拍 openmm-ml 装了才跑）。

### 插件形态（plugin plan-schema namespace 之上）

- ml_region 不消费插件机制解锁的能力（方法分发、注册表词汇表）；它消费的
  是 KernelSpec 装配通道——barostat/dummy_exceptions 先例所在；
- ML/MM 耦合规则（何时移除哪些 MM 项）是核心物理知识，放核心树内才能受
  source-scan 与固定决策纪律约束。

### fake 内核数值评估 mock NNP

- fake 无 MM 力可嵌入、零力传播，复刻 mock 势只加镜像维护面；CI 全管线
  保障由 openmm 适配器 + mock（无需 torch）承担，覆盖更强。

## 后果

### 正面

- ML/MM 全管线（含机械嵌入与力组分配）在默认 CI 门内永久可测，零 torch；
- 自有模型 .pt 即接口，接任何 TorchScript 势（自有 NNP、公开模型）无需改
  核心；
- openmm 版本分叉被显式圈死在 `ml` 环境并留有升级钩子。

### 负面（已认领）

- `ml` 环境临时用 openmm 8.5.2（与生产 8.6 并存至 conda-forge 追平）；
- 机械嵌入移植体是 openmm-ml 的快照（`501c3a0` / v1.7），上游后续修正需
  人工同步（attribution 头记录来源与差异）；
- TorchScript 模型的单位契约（nm in / kJ/mol out）无法机器验证，只能文档
  化 + 测试示范。

## 重开条件

conda-forge 发布 tracking openmm 8.6+ 的 openmm-torch；或残基 ML 区附录落地跨边界
残基 ML 区（需要 link-atom / 电荷再分布语义，届时嵌入层需扩展）；或真
QM/ORCA 重启（另立 ADR）。

---

## 附录：活性位点残基 ML 区（跨界键处理，2026-09-03）

残基 ML 区附录落地时追加；上文其余章节不变。

### 决策

1. **ML 区从 ligand-only 扩展到残基选择器**：`ml_region.residues`
   接受 `"CHAIN:RESID"`（尾部数字 → 按 residue id 匹配，PDB 作者编号）
   与 `"CHAIN:NAME"`（尾部非数字 → 按 residue name 匹配，如配体
   `"A:JZ4"`）两种拼写，大小写不敏感；`indices` 与 `residues`
   **互斥**（两种方式同时定义会留下静默过期的索引表）。选择器在
   `neomd validate --check-files` 层与 openmm 适配器装配时各解析一次
   （后者是权威——手工构建的 `KernelSpec` 也走同一防御门）。语法与
   解析实现在 `neomd/ml/selection.py`（openmm-free，鸭子类型拓扑）。
2. **跨界键政策（a）——跨界 MM 键合项保留在 MM**：凡键/角/二面角
   含**任一** MM 原子即保留为 MM 项；只有全 ML 项从 MM 删除、交由
   NNP。依据：openmm-ml 的 `removeBonds` 本就只删全 ML 键（移植体
   行为逐字如此），也是 GROMACS QM/MM 共价边界的惯例。后果（诚实
   认领）：边界 ML 原子对 MM 伙伴仍带 MM 键合项，交界处自身化学由
   双方各自描述（机械嵌入、无 link atom、无电荷再分布）；
   `constraints: HBonds` 下 ML 区内 X-H 约束不删（约束无能量、
   不双计，但该自由度保持刚性）。link-atom 加帽与边界参数重拟合
   **列为后续工作**，与真 QM/MM 一并决策（届时另立 ADR）。
3. **非键例外同原逻辑**：ML-ML 对加零化例外；跨界 MM-MM/ML-MM 的
   预存 1-2 例外原样保留（不双计证明见
   `tests/v2/test_mlmm_residues.py` 的边界矩阵测试——存活/删除项
   与解析能量双向钉死）。
4. **真 QM/MM（ORCA / link atom / 电荷位移）维持暂缓**（卷首决策），
   重启时另立 ADR；本附录不为其预留任何接口。

### 演示

`examples/mlmm_ligand/run_mlmm.py --region active-site`：JZ4 配体 +
口袋残基（GLN102、LEU133，按晶体坐标 0.26/0.36 nm 选定）为 ML 区，
min + MD 两腿；mock 层默认门内可跑，torch 层在 `ml` 环境跑 toy 模型。
