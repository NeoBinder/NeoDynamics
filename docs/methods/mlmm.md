# ML/MM 耦合（ml_region：ligand-only + 活性位点残基）

> 需求：[issue #12](https://github.com/NeoBinder/NeoDynamics/issues/12) ·
> 决策（含活性位点残基的跨界键附录）：
> [ADR-0004](../adr/0004-mlmm-in-tree-coupling.md)

## 原理简述

additive ML/MM：体系的一部分（配体或活性位点残基）由 NNP
（机器学习势）描述，其余保持 MM。装配采用 openmm-ml 的机械嵌入
（ML-ML 的 MM 键合项与非键相互作用移除，ML 原子 MM 点电荷保留——
继续承担 ML↔MM 静电；无电荷再分布、无 link atom）。残基区的跨界
MM 键合项保留在 MM 侧（GROMACS QM/MM 共价边界惯例）——细节与依据
见 ADR-0004 附录。真 QM/MM（ORCA / link atom）暂缓，重启另立 ADR。

## 使用

一个 plan 段落把一个区域变成 ML 势区域；`indices` 与 `residues`
**互斥**：

```yaml
ml_region:
  indices: [1234, 1235, 1236]        # 0 基粒子索引（或 "1234,1235,..."）
  # 或 residues: ["B:JZ4", "A:102", "A:133"]   # "CHAIN:RESID"（作者编号）
  #                                            # 或 "CHAIN:NAME"（如配体）
  model:
    type: torchscript                # 或: mock
    path: my_nnp.pt                  # torchscript：模型文件即接口
    long_range_electrostatics: false # 周期体系必须声明
    periodic: true                   # 可选；默认随体系
    # mock 专属参数：tether_k (500 kJ/mol/nm^2)、repulsion_k (1 kJ/mol)、
    #               repulsion_sigma (0.15 nm)
```

- 校验走 collect-all（yaml key path + did-you-mean）；`neomd validate
  plan.yaml --check-files` 额外检查 indices 界内、residues 解析
  （未命中以 did-you-mean 报错）与 path 存在性。
- **模型单位契约**（必须精确）：模型收到**整个体系**的坐标
  （`float32`，`(N, 3)`，**nm**；ml_region 索引需烘焙进 forward
  内部），周期体系另传 nm 盒向量 `(3, 3)`，返回标量能量 **kJ/mol**；
  Å/eV/kcal 训练的模型在 forward 内换算。
- `long_range_electrostatics`：周期体系下的声明——`false`（默认）=
  ML-ML 非键直接移除；`true` = ML-ML 库仑按真实电荷乘积保留。
- 端到端示例：[examples/mlmm_ligand](../../examples/mlmm_ligand)
  （3HTB + JZ4；`--region active-site` 演示残基区）。
- torch 层测试：`pixi run -e ml test-ml`（默认门保持 torch-free；
  mock NNP 无 torch 即可走通全管线）。`ml` 环境临时 pin
  `openmm = "8.5.*"` + `openmm-torch`（升级 pin 是显式事件）。

## 产物

无专属 artifact；NNP 力参与常规能量/轨迹报告。装配发生在 openmm
适配器内、Context 创建前（NNP Force 不可 XML 序列化，不进
`system.xml`）；fake 内核忽略 `ml_region`。

## 参考文献

- [openmm-ml](https://github.com/openmm/openmm-ml)（MIT）——机械嵌入
  逐字移植源（v1.7 / commit `501c3a0`，attribution 在
  `ml/embedding.py`）；不作为依赖，仅 import-gated 交叉验证。
- REANN ML/MM, *JCTC* 2025 —— ML/MM 酶位点可用性（issue #12 引用）。
- [ADR-0004](../adr/0004-mlmm-in-tree-coupling.md) ·
  [issue #12](https://github.com/NeoBinder/NeoDynamics/issues/12)。
