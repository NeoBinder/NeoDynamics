# 增强采样分析与收敛诊断工具链（neomd.analysis）

> 状态：issue #16 · 已实现（W1-a，`src/neomd/analysis/` + `neomd analysis` CLI）· ADR：见
> [docs/adr/0001-neomd2-strangler-migration.md](../adr/0001-neomd2-strangler-migration.md)（v2 迁移总 ADR）

## 背景与动机

v1 的分析面只有 `bin/gethill.py` 与 `bin/hills_ana.py` 两个基础脚本（hills 求和 / 投影
FES）。issue #16 指出生产使用还缺五类能力：

1. **收敛判据** —— FES 随模拟时间的差值曲线（"FES 是否还在变"是停止采样的客观依据）；
2. **误差估计** —— block averaging 的统计误差条；
3. **rewighting** —— WT-MetaD 的 Tiwary–Parrinello reweight，从偏置轨迹恢复无偏期望值；
4. **多 walker 合并** —— 多副本共享 bias 目录时的合并分析；
5. **动力学** —— flooding / infrequent MetaD 的过渡时间统计。

更重要的是 v2 决策 #6 的有意破坏：新 artifact 格式（`colvar.tsv` / `hills.npz` /
`smd.tsv`）**有意**打破了 v1 `gethill` / `hills_ana` 的读者，"rewrite lands in
2.x" 点名的就是本工作项。v2 分析工具链必须读新格式，且是 GaMD reweight（#10）、OPES
（#11）、RBFE BAR/MBAR（#8）与 ML-CV 收敛诊断（#9）的共享基座，issue-dev-plan 因此将其
列为优先。

## 与 issue 方案的差异（v2 决策）

- 落点为 **`neomd.analysis` 子包 + `neomd analysis` CLI 子命令**，不再新增 `bin/fes_ana.py`
  之类的 bin/ 脚本 —— 与 v2 "facade + CLI 子命令" 的入口纪律一致。
- 读取对象是 v2 新 artifact（`colvar.tsv` / `hills.npz` / `smd.tsv` + `manifest.json`
  的 grid 元数据），**不做 v1 兼容**（决策 #6：无永久兼容层）。
- **不做绘图**：输出 tsv/json 到 stdout 或 `--out` 文件，numpy-only、确定性、openmm-free。
- flooding / infrequent-MetaD 动力学分析为**有记录的后续项**：新格式尚未定义该观测量，
  issue 中该项不在本次落地范围。
- 同一面可 `from neomd.analysis import fes_from_hills, block_average,
  reweight_expectation, ...` 供程序消费 —— 它是其它 method track 的基座。

## 使用

README 的
["Analyzing runs"](https://github.com/NeoBinder/NeoDynamics#analyzing-runs) 一节给出命令组摘要，命令组：

```bash
neomd analysis fes run_dir --out fes.tsv          # WT FES（与运行自身的 fes.tsv
                                                  # 同布局；--bins N 自定分辨率）
neomd analysis convergence run_dir --blocks 4     # 窗口切分 max/mean |dFES| 表
neomd analysis block-average run_dir --column phi # 某一 tape 列的均值+统计误差
                                                  # （也直接接受 .tsv 文件）
neomd analysis reweight run_dir --observable phi --cv phi --fes-out rw_fes.tsv
                                                  # Tiwary-Parrinello c(t) reweight
neomd analysis merge walker_a walker_b --out merged
                                                  # 多 walker hills 合并成一个 run 目录
```

约定要点：

- FES 估计器即生产者自身的 well-tempered 关系 `FES = -((T+dT)/dT) * bias`，
  `dT = T*(biasFactor-1)`；ledger 回放与运行中 bias **逐位一致**（测试钉死）。
- `hills.npz` 位置用 kernel 单位（角 CV 为弧度），`colvar.tsv` 用自然单位（度）——
  分析通过运行所用的同一 port 表换算。
- reweight 不需要 tape 里的 bias 列：`c(t)` 由每条 colvar 行**之前**存入的 hills
  重建（探针先于 deposition 触发，即该行实际采样所受的 bias）。

## 架构与产物

- 模块：`src/neomd/analysis/`，openmm-free，直接消费 sink 产物与 manifest 元数据。
- 产物：tsv/json（`--out`），无图形输出。
- 测试（`tests/v2/test_analysis.py` + `test_analysis_cli.py`）：**合成双势阱解析解**
  —— 手工布点的 hills 携带闭式 bias/FES，钉死网格回放、逐点求值、周期回绕、自定义
  bins；block averaging / reweight / 多 walker 合并各有公共接口测试（只跨公共接口，
  不探测内部）。

## 参考文献与 ADR

- Tiwary & Parrinello, *JPCL* 2015（c(t) reweighting）；Bussi group sum_hills /
  PLUMED 分析文档。
- ADR-0001（strangler 迁移）；`docs/v2-migration-plan.md` 决策 #6（artifact 破坏性
  变更、"rewrite lands in 2.x"）；`docs/issue-dev-plan.md` #16 行与 W1-a 行。
