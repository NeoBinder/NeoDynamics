# 机器学习集体变量（ML-CV，期 1：featurize → train → convert）

> 需求：[issue #9](https://github.com/NeoBinder/NeoDynamics/issues/9)（两期
> 拆分）· 期 2 注入设计：[ADR-0006](../adr/0006-mlcv-injection-torchcv.md)

## 原理简述

"短轨迹 → 训练 ML-CV → 驱动 OPES/MetaD" 闭环的出树工具（numpy-only，
零模拟核心改动）：特征化复用公开 CV 注册表的 `evaluate` 实现；训练层
是线性 TICA（慢线性分量，广义特征问题；runs 池化但不跨边界拼接）与
logistic 回归（两盆地标签）；`convert` 把线性模型导出 TorchScript
（torch-gated），与 numpy 侧 `apply_model` 逐位对拍——期 2
（TorchScriptCV 注入，ADR-0006）的交接产物。

## 使用

```bash
# 1. featurize：对 run 目录的 output.dcd 帧计算命名特征列
#    （距离、kind-driven coordination/path/rmsd CV、平滑接触数、tape
#    直通列），质量取自该 run 的 system.xml——确定性 npz 缓存
neomd mlcv featurize featurize.yaml            # run_dirs + features: {...}

# 2. train：无标签流的 TICA（lag τ 下慢线性分量）或两盆地标签的
#    logistic 回归
neomd mlcv train features.npz --model tica --lag 10 -o model.npz
neomd mlcv train features.npz --model logistic --label-column s --label-threshold 0

# 3. convert：线性模型导出 TorchScript（torch-gated）
neomd mlcv convert model.npz -o cv.pt
```

配置问题走 collect-all（key path + did-you-mean）。

## 产物

- `features.npz`——特征缓存；
- `model.npz`——版本化线性模型（TICA / logistic）；
- `cv.pt`——TorchScript 导出（float64），期 2 交接工件。

## 参考文献

- mlcolvar, *J. Chem. Phys.* 159, 010901（2023）——
  [arXiv:2305.19980](https://arxiv.org/abs/2305.19980)（社区标准栈的
  参照；不进依赖）。
- Trizio & Parrinello, *J. Phys. Chem. Lett.* 12, 8621（2021）——
  [Deep-LDA](https://doi.org/10.1021/acs.jpclett.1b01842)。
- [ADR-0006](../adr/0006-mlcv-injection-torchcv.md)（期 2 注入设计）·
  [issue #9](https://github.com/NeoBinder/NeoDynamics/issues/9)。
