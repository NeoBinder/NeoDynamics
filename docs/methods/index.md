# 方法文档索引

每个特性一份独立方法文档；本文件只做索引。

- [mlcv.md](mlcv.md) —— ML 集体变量期 1：featurize/train/convert CLI
  （issue #9 期 1，W2-c，ADR-0006）
- [opes.md](opes.md) —— OPES 增强采样方法（KDE bias、kernels.npz 回放；
  issue #11，W2-a，路径 B 自研）
- [analysis.md](analysis.md) —— 增强采样分析与收敛诊断工具链（`neomd.analysis` 子包 +
  `neomd analysis` CLI；issue #16，W1-a）
- [cv-library.md](cv-library.md) —— CV 库扩展：path（s,z）/ coordination /
  rmsd-as-CV 知识三元组与双轨实现（issue #14 残留，W1-b）
- [qc.md](qc.md) —— 结构质量校验 `neomd.qc`：openmm-free 几何检查 +
  prepare/min 挂钩 + `qc_report.json` + issue #7 回归（issue #15 + #7，W1-c）

- [boresch.md](boresch.md) —— Boresch 取向锚 restraint triple：3+3 锚原子、
  6 分量取向约束（issue #8 先行切片，W1-d，ADR-0003）
