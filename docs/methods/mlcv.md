# 机器学习集体变量（ML-CV，期 1：featurize → train → convert）

- issue：[#9](https://github.com/NeoBinder/NeoDynamics/issues/9)（机器学习
  集体变量与 ML 增强采样）
- 实现状态：期 1 已落地（出树工具、零核心改动）；期 2 注入见
  [ADR-0006](../adr/0006-mlcv-injection-torchcv.md)
- 决策记录：[ADR-0006](../adr/0006-mlcv-injection-torchcv.md)

## 背景与动机

README roadmap 提到 machine learning-powered MD。issue #9 指出：此前
CV 库只有 5 种手写几何 CV（distance /
distance_ref / min_distances / dihedral / angle），而结合-解离采样对
CV 质量高度敏感——文献反复证明简单距离 CV 在 trypsin-benzamidine 上
即失效，需要口袋水合、配体取向、接触图等复杂 CV。社区标准做法是
mlcolvar 栈：PyTorch 训练 Deep-LDA / Deep-TICA / SPIB → 导出
TorchScript → 注入 PLUMED/OpenMM 驱动偏置。目标是提供"短轨迹 →
训练 ML-CV → 驱动 OPES/MetaD"的完整闭环（与 OPES issue 联用）。

issue 技术方案四层：特征化层（Cα 距离矩阵、残基接触图、配体-口袋
接触数、口袋水合数）、训练层（mlcolvar 系模型 + TorchScript 导出）、
注入层（openmm-plumed `PYTORCH_MODEL` 或 OpenMM TorchForce）、迭代
流程（`bin/train_mlcv.py` 采样-训练循环 CLI）。

## 与 issue 方案的差异

issue #9（2026-09-02）拆两期，依据是真实的架构事实：
`CVIR` 是表达式字符串驱动（`expression` 是 Lepton 可编译的物理），
TorchScript 模型不是表达式——注入需要新的 CV kind，属 port 扩展。

- **期 1 是出树工具、零核心改动**：`neomd mlcv`
  CLI 子命令（不再是 `bin/train_mlcv.py` 脚本）做 featurize / train /
  convert 三步，numpy-only，不进模拟内核路径。mlcolvar 也不进依赖——
  首期模型是自实现的线性 TICA（慢线性分量，C_tau v = λ C_0 v 广义
  特征问题）与 logistic 回归（两盆地标签）。
- **期 2（注入）前置 ADR**：[ADR-0006]
  (../adr/0006-mlcv-injection-torchcv.md) 设计了新 CV kind
  `"TorchScriptCV"`（kind-driven CVIR，RMSD/coordination/PathCV
  先例）+ openmm 适配器 TorchForce-as-inner-CV（PathCV 组合的推广）+
  fake kernel torch eval-mode 确定性求值（双轨纪律 #5）。分层落地：
  切片 A 线性模型 expression 化（零新依赖端到端闭环），切片 B 通用
  TorchScript 非线性路径。issue 方案中的 openmm-plumed 后端被否决
  （绕过 KernelPort seam，与 #11 路径 A 被拒同理）。

## 使用

README 的「ML collective variables (phase 1)」一节是入口摘要；完整
三步 CLI：

```bash
# 1. featurize：对 run 目录的 output.dcd 帧计算命名特征列
#    （距离、kind-driven coordination/path/rmsd CV、平滑接触数、tape
#    直通列），质量取自该 run 的 system.xml——确定性 npz 缓存
neomd mlcv featurize featurize.yaml            # run_dirs + features: {...}

# 2. train：无标签流的 TICA（lag τ 下慢线性分量，runs 池化但不跨
#    边界拼接）或两盆地标签的 logistic 回归
neomd mlcv train features.npz --model tica --lag 10 -o model.npz
neomd mlcv train features.npz --model logistic --label-column s --label-threshold 0

# 3. convert：线性模型导出 TorchScript（torch-gated）——期 2 交接
#    产物，与 numpy 侧 apply_model 逐位对拍
neomd mlcv convert model.npz -o cv.pt
```

配置问题走 collect-all（key path + did-you-mean）。

## 架构与产物

- **模块**：`src/neomd/mlcv/`（featurizer + 训练 CLI + TorchScript
  导出）。特征化只经公开 CV 注册表的 `evaluate` 实现（不重复实现
  几何）；一切跨公开接口，无核心改动。
- **产物**：
  - `features.npz`——特征缓存（支持大轨迹的确定性缓存）；
  - `model.npz`——版本化线性模型（TICA / logistic），json header +
    numpy 数组；
  - `cv.pt`——`torch.jit.script` 的精确线性权重 TorchScript 导出
    （float64），与 numpy `apply_model` 逐位对拍；期 2 的交接工件。
- **torch 门**：convert 需要 torch（import-gated）；featurize/train
  全 numpy。

## 参考文献与 ADR

- [ADR-0006：ML-CV 期 2 注入——TorchScriptCV port 扩展（TorchForce
  组合）](../adr/0006-mlcv-injection-torchcv.md)
- [issue #9](https://github.com/NeoBinder/NeoDynamics/issues/9)（ML-CV
  两期拆分）
- mlcolvar, *JCP* 2023（arXiv:2305.19980）
- Trizio & Parrinello, *JPC Lett* 2021（Deep-LDA）
- Bonati & Parrinello, *JCTC* 2024（SPIB）
