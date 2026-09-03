# Issue 开发计划 —— v2 架构适配版（2026-09-02）

把 GitHub 上全部 11 个 issue（#7 已关闭，#8–#17 开放）整理成可并行执行的
开发计划。所有 issue 写于 v1 时代，引用的路径（`metadynamics/engine.py`、
`restraints/constructor.py`、`generic.Pipeline`、`bin/*.py`、
`qmmm/pipeline.py`）在 v2 中已不存在或已换形态，本计划逐条给出 v2 落点。

v2 架构事实（已核实，作为映射依据）：

- CV 与 restraint 都是知识三元组：`colvars.py` 现有 5 个 CV
  （distance / dihedral / angle / min_distances / distance_ref）；
  `restraints.py` 已有 9 个 restraint，**含 `funnel`（v1 全参数移植：
  width/steepness/s_center/buffer/lower/upper walls）和 `rmsd`**。
- 方法经 registry method rack 分发：`methods/metadynamics.py`、
  `methods/smd.py`，prepare 契约 + `driver.run_prepared_method`。
- `kernel/port.py` 已具备：`energy_forces()`（势能读出）、`GroupEnergy`
  capability（分组能量）、`BiasOps`（cv_values/bias_energy/update_table）、
  `BiasParamOps`（参数在线更新，SMD ramp 同款 seam）。
- GaMD 插件机制 drill 已验证（`examples/gamd_drill/`：注册/发现/分发三项，
  但只是机制验证，非真实 GaMD）。已知缺口：plugin plan-schema namespace
  （drill 用 `meta_set` ride-along 是临时方案）。
- CI 已建成（`pixi run test` + `test-golden` + 3HTB smoke + pre-commit
  check-only hooks）；tests/v2 460 passed。
- 冻结决策约束（AGENTS.md §Settled decisions）：#6 分析读旧格式属有意破坏、
  2.x 重写；#7 qmmm 以插件重建、双真实适配器；#8 多腿编排 deferred to 2.x。

研究决定（2026-09-02，openmm-ml 评估）：**openmm-ml 不作为核心依赖，自研
轻量 ML/MM 耦合模块**。依据：其通用层仅约 1.5k 行（`mlpotential.py` +
`embeddings/`），其余约 65 KB 是 9 个逐模型适配器（aimnet2/ani/mace/
nequip/deepmd/orb/torchmdnet/fennix/ase）——本计划要接**自有** ML potential，
这层整体用不上；openmm-ml 维护并不慢（2025-03→2026-06 发 1.3→1.7 五版，
最后推送 2026-08-29），但活跃 churn 恰集中在模型注册表层。真正绕不开的底座
是 **openmm-torch**（TorchForce 的 C++ 插件）+ torch——自研也省不掉，按
决策 #10 的 pin 纪律管理。自研内容 = 机械嵌入耦合逐字移植（openmm-ml
`mechanicalembedding.py` + `utilities.py`，MIT，带 attribution，同"物理
逐字移植"纪律）+ 一个通用 TorchScript 模型加载器（自有模型一等公民）；
openmm-ml 降级为可选交叉验证参照（装了才跑的 marker-gated 对拍测试）。

---

## 一、Issue 逐条 v2 适配判定

| Issue | 标题 | v2 判定 | 剩余工作（v2 落点） |
|---|---|---|---|
| #7 | minimize 异常坐标（已关闭） | v1 scipy-minimize 路径 bug；v2 min 走 port（openmm adapter 用 OpenMM minimizer） | 无独立工作项；复现输入转为 #15 的 QC 回归用例 |
| #8 | RBFE 模块 | 大改写。`builder`→`tools/`；`restraints/constructor.py`→restraint triple；`generic.Pipeline`→`drive()`/`PreparedMethod`；`bin/rbfe_ana.py`→`neomd` CLI 子命令 | Boresch restraint triple（可先行）；softcore 扰动的 port 扩展 ADR；λ 窗口编排（多腿编排决策 #8 的第一个真实客户）；BAR/MBAR 分析（依赖 #16） |
| #9 | ML-CV 与 ML 增强采样 | 分两期。期 1（featurizer + 训练 CLI）是出树工具零核心改动；期 2（注入）受阻于架构：`CVIR` 是表达式字符串驱动，TorchScript 模型 CV 需要新 CV kind | 期 1 可开工；期 2 前置 ADR（port 扩展：openmm TorchForce + fake kernel 的 torch evaluate，满足双轨规则决策 #5） |
| #10 | GaMD | 部分完成：插件 seam drill 已验证 | 前置：plugin plan-schema namespace（v2-dag follow-up，与 #12 共享）；boost 标定用 `port.energy_forces()`（已有）；LiGaMD 组 boost 用 `GroupEnergy` + `pick_free_force_group`；参数更新用 `BiasParamOps`；reweighting 依赖 #16 |
| #11 | OPES | 重定路径：**路径 B（自研）为主**。`methods/opes.py` 与 metadynamics triple 完全同构（KDE→表格→`update_table`），fake kernel 可确定性跑 OPES 数学；cyrushu 的论文摘要评论（2026-07-22）直接作为实现规格。路径 A（openmm-plumed）绕过 KernelPort seam，与架构相抵，降级为旁路实验 | `methods/opes.py`（standard + explore 两 mode）；`kernels.npz` 新 artifact + resume ledger 回放（镜像 `_replay_ledger`）；多 walker 编排（与 #8 的迷你编排共享） |
| #12 | QM/MM 修复 | **不修 v1 代码**（legacy `qmmm/pipeline.py` 是 WIP 非 bug，不在 bug-fix-only 范围）。**范围决定（2026-09-02）：真 QM/MM（ORCA 后端、link atom、电荷位移）暂缓；ML/MM 转入正式开发，且不走 openmm-ml 依赖（见卷首研究决定），自研耦合模块 + 自有模型加载器** | ML/MM 落点：`KernelSpec` 新增 `ml_region` 字段（`{"indices", "model", ...}`，走 barostat/dummy_exceptions 同款先例——openmm 适配器在 Context 创建前实现，fake 忽略或走 mock NNP）；NNP Force 不可 XML 序列化，所以**不能**在 prepare 层改 system.xml，必须在适配器内装配；自研内容：机械嵌入耦合（自 openmm-ml MIT 移植，带 attribution）+ 通用 TorchScript 加载器（openmm-torch TorchForce，自有模型一等公民，openmm-ml 的逐模型注册表不要）；两适配器纪律：生产 = openmm-torch + 自有/公开 TorchScript 模型，CI = 解析 mock NNP（无 torch 跑通管线）；首期 ligand-only ML 区，活性位点残基跨界处理二期；真 QM/MM 留待重启 |
| #13 | CI 基线 | **主体已完成**（v2 迁移 Wave 0 已建：ci.yml、断言测试、import 级覆盖、pre-commit） | 残留小件：ruff（scope 排除 `src/neomd_legacy` 与 `bin/`，check-only）、codecov 可选、GPU nightly 可选、README badge。建议 re-scope 后缩小或关闭 |
| #14 | CV 库 + funnel | **funnel 已存在**（v1 移植，参数齐全）；rmsd 已是 restraint 形态 | 剩余：`path`（s,z）/ `coordination` CV triples + rmsd 转为 CV 形态；每个 CV 双轨实现（force 表达式 + numpy evaluate，决策 #5）+ 手写几何对拍测试 |
| #15 | 结构质量校验 QC | v2 化：QC 模块应为 openmm-free（纯 numpy 几何 + SystemBundle 数据），不经 port | 新模块（`neomd.qc` 或 `prepare.py` 侧钩子）：clash（PBC 感知）/键长键角/配体几何/体系级检查；挂 `prepare.py` 与 min 流程尾部；`qc_report.json` 经 sinks；fail-fast 走 collect-all 风格；#7 输入做回归；RDKit 设 optional extra |
| #16 | 分析工具链 | 即决策 #6 点名的 "rewrite lands in 2.x"：读 `colvar.tsv`/`hills.npz`/`smd.tsv` 新格式 | 新 `neomd.analysis` 子包 + `neomd analysis` CLI 子命令（不再用 bin/ 脚本）：收敛差值、block averaging、Tiwary-Parrinello reweight、多 walker 合并、flooding 动力学；合成双势阱解析解测试。**是 #10/#11/#8/#9 的共享基座，优先做** |
| #17 | 文档站 | v2 化：配置参考可半自动生成——`plan.py` 的 KNOWN_KEYS/schema + registry vocab（`CV_EXPRESSIONS` 注释自称 single source of truth for tests/docs） | mkdocs-material + Pages workflow（与 #13 残留共享 CI 基建）；各特性教程页随特性任务落地（AGENTS.md 文档纪律本就要求同任务带文档） |

v2-dag.md 的 Known follow-ups 尚无 issue 跟踪，纳入本计划轨道：
plugin plan-schema namespace（→ W0）、golden scenarios 脱离
`neomd_legacy` import（→ 期终清理前置）、deprecation 窗口结束时删除
`neomd_legacy` + `neomd2` alias + `migrate_v1`（→ 期终）、RESP2 真工具
验证、CUDA statistical tier。

---

## 二、依赖图

```
W0 基座
  infra-ruff(#13残留) ──────────────┐
  docs-skeleton(#17) ───────────────┼─→ 后续所有 PR / 文档持续轨道
  plugin-plan-schema(v2-dag) ──→ #10 GaMD, #12 QM/MM
  golden-脱离legacy(时间触发) ──→ 期终清理

W1 共享基座
  analysis(#16) ────────→ #10 reweight · #11 FES · #8 BAR/MBAR · #9 收敛诊断
  cv-triples(#14残留) ──→ #9 featurizer 复用 coordination · #11 trypsin CV
  qc(#15 + #7回归) ─────→ 独立
  boresch + RBFE选型ADR(#8切片) →→→ W3 RBFE 引擎

W2 方法层（每方法一文件，天然并行）
  opes(#11, 路径B) ← 软依赖 cv-triples；测试用 analysis
  gamd(#10, 真插件) ← plugin-schema, analysis
  mlcv-期1(#9 featurizer+训练CLI) ← cv-triples；注入 ADR 同步做
  mlmm(#12 拆分后) ← KernelSpec.ml_region 扩展（入 port 扩展批次）；机械嵌入耦合自研移植 + TorchScript 加载器 + mock NNP；openmm-torch/torch pin 进 pixi；demo = min + 100 ps

W3 集成层
  rbfe(#8) ← boresch + analysis + λ编排 + port softcore ADR
  mlcv-期2(#9 注入) ← 注入 ADR + port 扩展
  mlmm 进阶(#12) ← 活性位点残基 ML 区（跨界边界处理）；真 QM/MM 暂缓，重启另立 ADR
  各特性教程页(#17 持续) + 期终清理（时间触发）
```

## 三、波次与并行轨道

并发度建议 3–4 条并行轨道（每轨道 = 一个 `.worktrees/<name>` worktree，
squash 落 main，遵守 AGENTS.md 完成门槛）。

### Wave 0 —— 基座（小而快，先合入）

| 轨道 | 内容 | 来源 |
|---|---|---|
| W0-a | ruff 接入 pre-commit（排除 legacy/bin）、README CI badge；codecov/GPU nightly 决定做不做 | #13 残留 |
| W0-b | mkdocs-material 骨架 + Pages workflow + 配置参考半自动生成（plan.py schema + registry vocab 表） | #17 切片 |
| W0-c | plugin plan-schema namespace：ADR + 实现（drill 的 `meta_set` ride-along 替换为正式机制） | v2-dag follow-up，解锁 #10/#12 |

### Wave 1 —— 共享基座（全并行，互不碰文件）

| 轨道 | 内容 | 来源 |
|---|---|---|
| W1-a | `neomd.analysis` 子包：colvar.tsv/hills.npz 读取、FES 收敛差值、block averaging、TP-reweight、双势阱解析解测试；`neomd analysis` CLI | #16 |
| W1-b | `colvars.py` 新增 path / coordination triples + rmsd-as-CV；双轨实现 + 几何对拍测试 | #14 残留 |
| W1-c | openmm-free QC 模块 + prepare/min 挂钩 + `qc_report.json` + #7 回归用例 | #15 + #7 |
| W1-d | Boresch restraint triple + RBFE 技术选型 ADR（openfe/openmmtools vs 自研 softcore） | #8 切片 |

### Wave 2 —— 方法层（每方法独立模块，天然并行）

| 轨道 | 内容 | 来源 |
|---|---|---|
| W2-a | `methods/opes.py`：KDE + 表格 bias（同 metadynamics seam）、standard/explore 双 mode、`kernels.npz` + resume 回放、fake kernel 确定性测试（cyrushu 评论为规格） | #11 |
| W2-b | 真实 GaMD 插件：标定预跑（`energy_forces` 统计 Vmax/Vmin/σ）、dual boost、LiGaMD 组定义（`GroupEnergy`）、reweight（用 W1-a） | #10 |
| W2-c | ML-CV 期 1：featurizer（复用 W1-b coordination）+ 训练 CLI + npz 缓存；同时产出期 2 注入 ADR（port TorchCV 扩展方案） | #9 |
| W2-d | ML/MM 实装：ADR（in-tree `KernelSpec` 扩展 vs 插件形态；openmm-ml 不依赖的决策记录）；`ml_region` spec + 机械嵌入耦合移植（自 openmm-ml，MIT + attribution）+ 通用 TorchScript 加载器（openmm-torch）；mock NNP 让 CI 无 torch 跑通管线；openmm-torch/torch 按 pin 纪律进 pixi.toml；ligand-only ML 区 demo（min + 100 ps MD）；openmm-ml 装了才跑的对拍测试（marker-gated） | #12（ML/MM 部分） |

### Wave 3 —— 集成层

| 轨道 | 内容 | 来源 |
|---|---|---|
| W3-a | RBFE 引擎：λ 窗口迷你编排（多腿编排 2.x 决策的第一客户）、softcore port 扩展、BAR/MBAR、CDK2/trypsin 基准 | #8 |
| W3-b | ML-CV 期 2：TorchScript 注入（port 新 CV kind + openmm TorchForce + fake torch evaluate）、迭代采样-训练闭环 | #9 |
| W3-c | ML/MM 进阶：活性位点残基 ML 区（键跨越边界的处理）；真 QM/MM 暂缓，重启时另立 ADR | #12 |
| 持续 | 各特性教程页、examples；期终：golden scenarios 脱离 legacy → 删 `neomd_legacy`/`neomd2`/`migrate_v1` | #17 + v2-dag |

### 优先顺序（业务视角）

1. **W0 全部**（小、解锁一切，避免中途加 ruff 造成在飞分支 churn）。
2. **W1-a analysis 优先于一切方法层**——四个特性 issue 都消费它；
   W1-b CV triples 次之；W1-c/W1-d 随并行额度。
3. **W2-a OPES 优先**（与 metadynamics 镜像、fake kernel 友好、规格已备齐、
   采样业务价值最高），W2-b GaMD 次之；**W2-d ML/MM 为已明确点名的开发项
   （2026-09-02 决定），按并行额度与 OPES/GaMD 同波推进**。
4. W3 各项链路最长（RBFE 尤甚），ADR 在 W1 就位，引擎放最后；真 QM/MM
   暂缓，不在本计划排期内。

## 四、并行冲突管控

- **`plan.py` KNOWN_KEYS / schema 是最高冲突文件**（每个方法/CV 都要动）：
  schema 变更做小 PR、先合入再开工特性主体；W2 各轨道开工前把各自
  schema section 一次性提入。
- **`kernel/port.py` 扩展集中管理**（#9 TorchCV、#12 `KernelSpec.ml_region`
  字段、可能 #8 softcore）：合并为一个"port 扩展"审阅批次或严格串行，避免
  seam 漂移；新增 seam 时同步扩展 source-scan 保证（AGENTS.md 工作纪律）。
- **`colvars.py`**：W1-b 先合入，W2-c 之后动它。
- **`methods/`** 每方法一文件，低冲突；analysis 是纯新目录。
- 测试一律走公共接口（md_run/compile/register/port ops），不探内部——
  新方法的测试照 `test_metadynamics.py`/`test_smd.py` 的形态写。

## 五、Issue 侧动作建议（GitHub 上同步）

- **#13**：评论说明 v2 迁移已达成主体，re-scope 为残留小件或完成 W0-a 后关闭。
- **#14**：更新正文——funnel 已存在（v1 移植），剩余为 path/coordination/rmsd-CV。
- **#11**：评论锁定路径 B 为主（架构理由：KernelPort 三适配器 + fake
  kernel 确定性），路径 A 降级为旁路实验；cyrushu 的规格评论置顶引用。
- **#12**：拆分——ML/MM 转入正式开发（`KernelSpec.ml_region` + 自研耦合
  模块 + TorchScript 加载器 + mock NNP，ligand-only 先行；不依赖 openmm-ml，
  理由见卷首研究决定），真 QM/MM（ORCA/link atom）标记暂缓（2026-09-02
  决定），删去"修复 v1 pipeline.py"的表述。
- **#8**：标注前置依赖（λ 编排 = 多腿决策 revisit、analysis、Boresch）。
- **新开 issue**：plugin plan-schema namespace、port 扩展 rack（TorchCV +
  QM capability）、deprecation 窗口清理（三项 v2-dag follow-ups 目前无跟踪）。
