# OPES（method: opes）

> **来源 issue**：[#11 [Feature] 支持 OPES 增强采样（WT-MetaD 的现代升级路径）](../issue-dev-plan.md)
> **实现状态**：已实现（`src/neomd/methods/opes.py`，Wave 2 轨道 W2-a，路径 B 自研）
> **关联 ADR**：无（自研决策记录在 issue 开发计划 §一 #11 行与下文"差异"节）
> **实现规格**：cyrushu 的 issue #11 评论（2026-07-22）+ Invernizzi–Parrinello 2020/2022 论文

## 背景与动机

OPES（On-the-fly Probability Enhanced Sampling，Invernizzi & Parrinello,
2020 —— issue 引用写作 *JPCL*，实现参照版本为 *JCTC* 16, 7113）通过 KDE 在线估计概率分布构造偏差，偏差准静态、参数更少
（PACE / BARRIER / SIGMA），并分化出 OPES-explore（耐次优 CV）、OPES-flooding
（结合/解离动力学）等变体，已是 PLUMED 生态的默认推荐。相对 well-tempered
metadynamics 的网格存储 bias，OPES 摆脱了"每次加 hill 更新整个网格、网格随
CV 维数指数增长"的结构性限制。

issue #11 写于 **v1 时代**：其正文描述的 `metadynamics/engine.py`（numpy 网格 +
`CustomCVForce` 查表插值）、`metadynamics/colvar.py` 的 `idstr2list`、
`bin/hills_ana.py` 分析链、`engine.py` 旁新增 `OPESEngine` 的落点，在 v2 中
均已重构（见 [docs/v2-migration-plan.md](../v2-migration-plan.md)）。v2 落点
（issue 开发计划 §一 #11 行）：

- **`src/neomd/methods/opes.py`**：方法知识三元组，与 metadynamics triple
  完全同构——KDE → 表格 → `update_table`，经 registry 分发，由
  `drive()` 的 prepare 契约调度；
- CV 定义复用 `colvars.py` 的知识三元组词汇表（不再有独立的
  `colvar.py`/PLUMED 语法转换层）；
- 分析经 `neomd.analysis`（issue #16 的新工具链），不再有 `bin/hills_ana.py`。

## 与 issue 方案的差异（v2 决策）

issue 给出两条路径，v2 **重定路径：路径 B（自研）为主，路径 A 降级为旁路
实验**（issue 开发计划 §一 #11 行，2026-09-02）：

- **路径 A（openmm-plumed / PLUMED `OPES_METAD` 驱动）被降级**：它绕过
  KernelPort seam 直接注入 OpenMM Simulation，与 v2 的三适配器架构
  （openmm / fake / replay，`provides()` 能力协商）相抵——fake kernel 的
  确定性测试与 resume 的 bit-exact 保证都会失去定义点。
- **路径 B（自研）为主**：`methods/opes.py` 与 metadynamics triple 完全
  同构（`BiasIR(kind="CustomCVTableForce")` 经 `install_bias` 安装，
  `on_step_interval = opes_set.pace` 驱动更新，`bias_ops().update_table`
  推表），fake kernel 可确定性跑全部 OPES 数学（KDE 无 RNG，压缩是
  最近核而非随机采样）。
- 实现规格 = **cyrushu 的 issue #11 评论（2026-07-22）**（论文摘要级公式，
  直接作为规格），细节以 Invernizzi–Parrinello 2020/2022 论文 + PLUMED
  参考实现（仅作文档参照，未复制代码）补齐。
- issue 任务清单中的 "examples ala OPES 示例 / trypsin-benzamidine 结合
  示例 / 多 walker 编排" 属后续工作（多 walker 与 #8 的迷你编排共享，
  见计划 §二 W2-a）。

## 使用

OPES 与 metadynamics 共用同一 facade：`method: opes` + 相同的 `colvars:`
段（网格即 bias 表定义域，各 CV 的 `biasWidth` 即初始核宽 σ(0)）+
`opes_set:` 段。`opes_set` 只收方法真正需要的三个输入——`pace`（偏差
更新间隔步数）、`barrier`（预期自由能垒，kJ/mol）、可选 `mode: standard`
（默认，收敛导向的 well-tempered 目标）或 `mode: explore`（均匀探索目标）。
γ、ε 与核截断均由 `barrier` 推导；**没有** `biasFactor`/`height` 键。

每 `pace` 步存入一个（压缩后的）KDE 核、刷新已探索区域的归一化 Z_n、
并经 metadynamics 同款 seam 推送新 bias 表。最小可运行 plan
（单个 distance CV）：

```yaml
method: opes
steps: 50000
temperature: 298
seed: 2026
integrator: {dt: 0.002, friction_coeff: 1.0}
input_files:
  complex: /work_dir/min/last.pdbx
  system: /work_dir/sys_prep/htb/system.xml
output:
  output_dir: /work_dir/opes
  trajectory_interval: 1000
  checkpoint_interval: 1000
colvars:
  dist:
    type: distance
    grp1_idx: "0"
    grp2_idx: "1"
    min_cv_nm: 0.2          # bias 表定义域
    max_cv_nm: 1.2
    biasWidth_nm: 0.05      # 初始核宽 σ(0)
    bins: 200
opes_set:
  pace: 500                 # 偏差更新间隔（步）
  barrier: 25.0             # 预期自由能垒，kJ/mol
  # mode: explore           # 或 standard（默认）
```

## 架构与产物

- **方法 triple**：`methods/opes.py` 注册为 `method: opes`（standard +
  explore 双 mode），镜像 `methods/metadynamics.py`——schema + 力表达 +
  observables 一个定义点，`entry.prepare(...) -> PreparedMethod` 后由
  DRIVER 跑循环。
- **数学要点**（cyrushu 规格）：无偏边际的加权 KDE
  `P_n(s) = Σ w_k G(s, s_k)/Σ w_k`，`w_k = exp(β V_{k-1}(s_k))`
  （explore 模式估计的是被偏置的采样分布，`w_k = 1`）；
  `V_n(s) = (1-1/γ)(1/β) log(P_n(s)/Z_n + ε)`（standard）、
  `(γ-1)(1/β) log(p^WT_n/Z_n + ε)`（explore）；Z_n 在**已探索**区域上
  平均（exit-time 修正）；自适应带宽 Silverman 收缩；最近核压缩
  （Mahalanobis 阈值内合并，递归重试）；核在 KERNEL_CUTOFF 个 σ 处截断。
  与 PLUMED 的已知偏离：explore 模式不把用户 SIGMA 放大 √γ（规格如此）。
- **tape / artifact**：`colvar.tsv`（CV 自然单位）、`kernels.npz`
  （核 ledger `{steps, positions, sigmas, heights, logweights}`，压缩前
  存入、即 resume 回放态）、`fes.tsv`（run 结束的 FES，estimator 按
  mode：standard `-(1/β)log P_n`，explore `-γ/β·log p^WT_n`）。
  `kernels.npz` 是在 deposit 钩子上写的方法 STATE（与 `hills.npz` 同款），
  **不是** switch-gated tape——probe 先于 `on_step` 触发，probe 写 ledger
  会滞后一次 deposit、破坏 bit-exact resume。
- **resume 语义**：`continue_md: true` 时 `resume.py` 裁剪 `kernels.npz`
  至 checkpoint 步，续跑经同一 deposit 数学回放 ledger，核与直跑
  bit-identical（同 metadynamics 的 hills 回放）。
- **与 analysis 的衔接**：`neomd.analysis` 读回 `colvar.tsv` /
  `kernels.npz` 做后处理（收敛差值、block averaging、TP-reweight、
  multi-walker merge）；flooding 动力学分析为已记录的后续项。

## 参考文献

- Invernizzi & Parrinello, *JPCL* 11, 2731（2020）—— OPES（issue 引用）。
- Ray & Parrinello, *JCTC*（2022）—— OPES-explore（issue 引用；实现参照的
  explore 目标分布出处为 Invernizzi, Piaggi & Parrinello, *JCTC* 18, 3988,
  2022）。
- PLUMED 文档：`OPES_METAD` / `OPES_METAD_EXPLORE` / `OPES_FLOODING`
  （issue 引用；实现期仅作文档参照，未复制代码）。
- [docs/issue-dev-plan.md](../issue-dev-plan.md) §一 #11 行、§三 W2-a ——
  路径 B 决策与轨道。
- [AGENTS.md](https://github.com/NeoBinder/NeoDynamics/blob/main/AGENTS.md) "Knowledge triples" 段 —— opes triple 的
  as-built 描述。
