# 方法文档索引

每页四段：**原理简述**（≤1 段，不复述论文推导）、**使用**（可运行
的 plan/CLI 示例）、**产物**（artifact 与 resume 语义）、**参考文献**
（实现依据的文献 + DOI/arXiv 链接）。设计决策与 issue 演进在 ADR 与
issue 里——只链接，不搬运。

- [gamd.md](gamd.md) —— GaMD 高斯加速 MD：标定预跑、dual/LiGaMD 模式
  （issue #10，ADR-0005）
- [opes.md](opes.md) —— OPES 增强采样：KDE bias、kernels.npz 回放
  （issue #11，自研）
- [rbfe.md](rbfe.md) —— RBFE λ 窗口：run_ladder、du.tsv 带、BAR/MBAR
  （issue #8，ADR-0003 / ADR-0007）
- [boresch.md](boresch.md) —— Boresch 取向锚 restraint：3+3 锚原子、
  6 分量取向约束（issue #8 先行切片，ADR-0003）
- [cv-library.md](cv-library.md) —— CV 库扩展：rmsd / coordination /
  path（s,z）知识三元组（issue #14 残留）
- [mlmm.md](mlmm.md) —— ML/MM 耦合：ml_region（indices + residues
  选择器）+ TorchScript/mock NNP（issue #12，ADR-0004）
- [mlcv.md](mlcv.md) —— ML 集体变量期 1：featurize/train/convert CLI
  （issue #9 期 1，ADR-0006）
- [analysis.md](analysis.md) —— 分析工具链：`neomd.analysis` 子包 +
  `neomd analysis` CLI（issue #16）
- [qc.md](qc.md) —— 结构质量校验 `neomd.qc` + `qc_report.json`
  （issue #15 + #7）
