# CV 库：path / coordination / rmsd（colvars.py W1-b 扩展）

> 状态：issue #14（残留部分）· 已实现（W1-b，`src/neomd/colvars.py` 新增 4 个
> kind-driven CV）· ADR：见
> [docs/adr/0001-neomd2-strangler-migration.md](../adr/0001-neomd2-strangler-migration.md)

## 背景与动机

issue #14 面向蛋白-配体**结合/解离**采样，要求把 v1 的 5 种 CV
（`distance` / `distance_ref` / `min_distances` / `dihedral` / `angle`）扩展为
CV 库 + funnel restraint。但 issue-dev-plan 对 #14 的再评估已判定现状：

- **funnel 已存在**（v1 移植的 `funnel` restraint triple，参数齐全）；
- **rmsd 已是 restraint 形态**。

因此剩余工作即本档记录的 **path（s, z）/ coordination CV triples + rmsd 转为 CV
形态**。这些是结合口袋水合描述（coordination）、路径采样（path）与构象对齐
（rmsd-CV）的高频刚需，v1 没有先例 —— 属于"来自一手文献的新物理"，不是 v1 移植。

## 与 issue 方案的差异（v2 决策）

- **知识三元组**：每个 CV 是 `colvars.py` 里的一个模块级条目（schema + kernel 表达式
  + numpy `evaluate` 观测量），经 `registry.register("cv", name, entry)` 注入 —— 不再
  改 v1 的 `metadynamics/colvar.py` 生成函数 + 注册表。
- **双轨实现（决策 #5）**：openmm adapter 编译真实力（RMSDForce、
  CustomNonbondedForce 对求和、逐参考帧 RMSDForce 的 log-sum-exp CustomCVForce）；
  fake kernel 携带镜像的 numpy 特殊路径，与 `colvars.evaluate` 逐位对拍钉死。这是
  fake/replay 无 OpenMM CV 求值所强制的。
- path CV 采用 Branduardi–Gervasio–Parrinello 定义并拆为 `path_s` / `path_z` 两个
  独立注册的 CV，共享同一 spec 块语法；`CVIR.kind` 驱动编译分发。
- funnel 不在本工作项（已存在，v1 移植、物理 verbatim）。

## 使用

README "Knowledge triples and the registry" 一节是摘要；各新 CV 的 YAML 拼写
（`colvars:` 列表项）：

```yaml
method: metadynamics
colvars:
  - cv_type: rmsd            # Kabsch 最优旋转 RMSD（nm 网格）
    ref_pos_file: ref.pdbx   # 全体系参考坐标（每粒子一行，nm）
    restr_grp: [10, 11, 12, 15]
    min_cv_nm: 0.0
    max_cv_nm: 1.0
    biasWidth_nm: 0.02
    bins: 50

  - cv_type: coordination    # 两原子团间的配位数（无量纲网格）
    grp1_idx: [10, 11, 12]
    grp2_idx: [40, 41]
    r0: 0.35                 # 参考距离（nm）
    nn: 6                    # 切换函数分子指数（默认 s(r)=1/(1+(r/r0)^6)）
    mm: 12                   # 分母指数
    min_cv: 0.0
    max_cv: 8.0
    biasWidth: 0.5
    bins: 40

  - cv_type: path_s          # 路径进度（无量纲网格）
    ref_path_file: path.pdb  # 多模型参考帧（MODEL/ENDMDL 或 pdbx_PDB_model_num，>=2 帧）
    restr_grp: [10, 11, 12]
    lambda: 0.35             # 平滑长度（nm），权重 exp(-MSD/lambda^2)
    min_cv: 0.0
    max_cv: 1.0
    biasWidth: 0.05
    bins: 50
  # path_z 同 ref_path_file/restr_grp/lambda，但为 nm 网格（min_cv_nm/...）
```

**numpy evaluate 对拍**：`neomd.colvars` 暴露 `evaluate` 观测量（Kabsch RMSD、
有理切换函数对求和、path 帧权重），fake kernel 的特殊路径与其逐位一致，由
`tests/v2/test_colvars_w1b.py` 用**手写几何值**对拍钉死（公共接口测试，不探测内部）。

## 架构与产物

- 注册表：`register("cv", "rmsd"/"coordination"/"path_s"/"path_z", ...)`，与 5 个
  v1 表达式 CV 并列，共 9 个 CV。
- kind-driven 编译：openmm 侧按 `CVIR.kind` 分发到 RMSDForce / pair-sum /
  log-sum-exp CustomCVForce；grid 约定 —— `rmsd`、`path_z` 用 nm 后缀
  （`min_cv_nm`/`biasWidth_nm`），`coordination`、`path_s` 无量纲（`min_cv`/`biasWidth`），
  沿用 v1 BiasVariable 的 gridWidth→bins 映射。
- 引用与 kernel 字符串记在 `colvars.py` 模块 docstring（一手文献、每个 CV 的
  kernel/表示约定）。

## 参考文献与 ADR

- Branduardi, Gervasio & Parrinello, *JCTC* 2007（path CV s,z）；PLUMED
  PATH / COORDINATION 文档；Limongelli et al., *PNAS* 2013（funnel metadynamics，
  背景）。
- ADR-0001（strangler 迁移）；`docs/v2-migration-plan.md` 决策 #2（v1 物理
  verbatim——新 CV 不适用，属新物理）与 #5（双轨 restraint/CV 报告）；
  `docs/issue-dev-plan.md` #14 行（"funnel 已存在，剩余为 path/coordination/rmsd-CV"）。
