# 方法文档索引

每个特性一份独立方法文档；本文件只做索引。

- [gamd.md](gamd.md) —— GaMD 高斯加速 MD：BoostOps 缝、标定预跑、
  dual/LiGaMD 模式（issue #10，ADR-0005）
- [mlmm.md](mlmm.md) —— ML/MM 耦合：ml_region（indices + 活性位点
  residues 选择器）+ TorchScript/mock NNP 两适配器（issue #12，ADR-0004）
- [mlcv.md](mlcv.md) —— ML 集体变量期 1：featurize/train/convert CLI
  （issue #9 期 1，ADR-0006）
- [opes.md](opes.md) —— OPES 增强采样方法（KDE bias、kernels.npz 回放；
  issue #11，自研）
- [rbfe.md](rbfe.md) —— RBFE λ 窗口：run_ladder、du.tsv 带、
  BAR/MBAR 分析（issue #8，ADR-0003 / ADR-0007）
- [boresch.md](boresch.md) —— Boresch 取向锚 restraint triple：3+3 锚原子、
  6 分量取向约束（issue #8 先行切片，ADR-0003）
- [cv-library.md](cv-library.md) —— CV 库扩展：path（s,z）/ coordination /
  rmsd-as-CV 知识三元组与双轨实现（issue #14 残留）
- [analysis.md](analysis.md) —— 增强采样分析与收敛诊断工具链（`neomd.analysis` 子包 +
  `neomd analysis` CLI；issue #16）
- [qc.md](qc.md) —— 结构质量校验 `neomd.qc`：openmm-free 几何检查 +
  prepare/min 挂钩 + `qc_report.json` + issue #7 回归（issue #15 + #7）
