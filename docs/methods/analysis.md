# 增强采样分析工具链（neomd.analysis）

> 需求：[issue #16](https://github.com/NeoBinder/NeoDynamics/issues/16) ·
> 迁移决策：[ADR-0001](../adr/0001-neomd2-strangler-migration.md)

## 原理简述

对新 artifact 格式（`colvar.tsv` / `hills.npz` / `smd.tsv` + manifest
grid 元数据）的 openmm-free 后处理工具链：WT FES 重建（估计器即
生产者自身的 well-tempered 关系 `FES = -((T+dT)/dT) * bias`，ledger
回放与运行中 bias 逐位一致）、FES 收敛窗口、block averaging 误差条、
Tiwary–Parrinello reweight（`c(t)` 由每条 colvar 行之前存入的 hills
重建——探针先于 deposition 触发，即该行实际采样所受的 bias）、
多 walker 合并。numpy-only、确定性、无图形输出；也是 GaMD reweight、
OPES、RBFE BAR/MBAR 等方法 track 的共享基座，可 import
（`from neomd.analysis import fes_from_hills, block_average, ...`）。

## 使用

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

`hills.npz` 位置用 kernel 单位（角 CV 为弧度），`colvar.tsv` 用自然
单位（度）——分析通过运行所用的同一 port 表换算。不做 v1 兼容
（`colvar.tsv`/`hills.npz`/`smd.tsv` 有意打破 gethill/hills_ana 读者）。

## 产物

tsv/json 到 stdout 或 `--out` 文件。flooding / infrequent-MetaD
动力学分析为有记录的后续项。

## 参考文献

- Tiwary & Parrinello, *J. Phys. Chem. Lett.* 6, 506（2015）——
  [c(t) reweighting](https://doi.org/10.1021/jz5013266)。
- [PLUMED / sum_hills 分析文档](https://www.plumed.org/doc)。
- [issue #16](https://github.com/NeoBinder/NeoDynamics/issues/16) ——
  需求；[ADR-0001](../adr/0001-neomd2-strangler-migration.md) —— 迁移。
