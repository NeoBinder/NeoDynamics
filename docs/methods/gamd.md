# GaMD（method: gamd）

> 需求与设计决策：[issue #10](https://github.com/NeoBinder/NeoDynamics/issues/10) ·
> [ADR-0005 GaMD boost 内核缝](../adr/0005-gamd-boost-seam.md)（BoostOps seam、
> 加性多通道缩放、全部被否决的替代方案）

## 原理简述

GaMD（Gaussian accelerated MD）对势能面加谐波 boost
ΔV(P) = ½·k·(E−P)²（P < E 时，否则 0），boost 力是缩放的系统力
F\* = −(1−k(E−P))·∇P，实现**无 CV** 增强采样——适合没有明确 CV 的场景
（口袋开合、loop 运动、配体进出路径），与 metadynamics 互补。
LiGaMD 对配体相关 force group 额外加一个通道，专门加速结合/解离。
 boost 参数 (E, k) 由无偏标定预跑按文献阈值/谐波两模式自动选出，
重加权 w = exp(βΔV) 由 `neomd.analysis` 提供。

## 使用

GaMD 换入 `method: gamd` 和一个 `gamd:` 段：`mode: total` 或 `dual`
（total + 二面角通道，扭转力自动隔离进独立 force group）；显式
`channels: [{label, groups}]` 定义覆盖 LiGaMD 式体系（系统 XML 中那些
相互作用已自成 force group）；`sigma0`（kJ/mol，默认 6.0）为 boost 强度
旋钮；标定预跑长度/间隔在 `calibration_steps` / `calibration_interval`
——boost 安装时零强度，`steps` 内先跑一小段无偏标定（`steps` 是**最终
步数**），按文献阈值/谐波对选出参数、写 `gamd_calibration.json` 并在线
推送。最小可运行 plan（total boost；restraint 墙只是给标定一点势能起伏）：

```yaml
method: gamd
steps: 500000              # 最终步数 —— 标定在其内部跑
temperature: 298
seed: 2026
integrator: {dt: 0.002, friction_coeff: 1.0}
input_files:
  complex: /work_dir/min/last.pdbx
  system: /work_dir/sys_prep/htb/system.xml
output:
  output_dir: /work_dir/gamd
  trajectory_interval: 1000
  checkpoint_interval: 1000
  report_gamd: true        # 写 gamd.tsv（默认开）
restraint:                 # 可选，建议标定前加
  dist: {type: distance, grp1: "0", grp2: "1", restr_k: 500.0, max_nm: 0.8}
gamd:
  mode: total              # 或 dual（total + 二面角通道）
  sigma0: 6.0              # kJ/mol，boost 强度旋钮
  calibration_steps: 50000 # 无偏预跑，含在 steps 内
  calibration_interval: 50 # Vmax/Vmin/σV 采样步长
  frequency: 10            # gamd.tsv 记录步长
  # LiGaMD 式：对预先分好的 force group 显式定义通道：
  # mode: channels
  # channels: [{label: ligand, groups: [3]}]
```

GaMD 不与 metadynamics 组合（安装 boost 后再 `install_bias` 直接报错）。

## 产物

- **`gamd.tsv`**：每通道 ΔV / 目标能量 P / 力缩放 s 的 boost trace
  （switch `output.report_gamd`，resume 时照常裁剪）。
- **`gamd_calibration.json`**：每通道 Vmax/Vmin/σV 样本与选定的
  (threshold, k)——唯一参数来源。
- **resume**：`continue_md: true` 不重新标定——从
  `gamd_calibration.json` 读回参数重推（幂等）。
- **reweighting**：`neomd analysis reweight`（w = exp(βΔV)）。

## 参考文献

- Miao, Feher & McCammon, *J. Chem. Theory Comput.* 11, 5208（2015）——
  [GaMD 原始论文](https://doi.org/10.1021/acs.jctc.5b00436)。
- Miao, *J. Chem. Phys.* 152, 244108（2020）——
  [LiGaMD](https://doi.org/10.1063/5.0005907)。
- Copeland, Miao 等, *J. Phys. Chem. B* 126, 481（2022）——
  [gamd-openmm](https://doi.org/10.1021/acs.jpcb.1c0864)（其 Langevin
  更新形式被移植；乘性双 boost 缩放被否决，见 ADR-0005）。
- [ADR-0005](../adr/0005-gamd-boost-seam.md) —— 设计决策全文。
