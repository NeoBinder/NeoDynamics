# ADR-0006：ML-CV 期 2 注入 —— TorchScriptCV port 扩展（TorchForce 组合）

- 状态：已提议（2026-09-03，W2-c 产出设计；W3-b「mlcv-期2」实装时确认）
- 决策者：项目维护者
- 关联：[issue 开发计划](../issue-dev-plan.md)（#9 两期拆分、W2-c/W3-b 轨道）、
  [W1-b CV triples](https://github.com/NeoBinder/NeoDynamics/blob/main/src/neomd/colvars.py)（kind-driven CVIR 先例）、
  W2-d ML/MM 轨道的 TorchForce 单位契约（见下「共享接口契约」）、
  期 1 实现 `src/neomd/mlcv/`（featurizer + 训练 CLI + TorchScript 导出）

## 背景

issue #9 拆两期。期 1（W2-c，本分支）是出树工具：`neomd mlcv
featurize/train/convert` 产出特征缓存（features.npz）、线性模型
（model.npz：TICA / logistic）与 TorchScript 导出（`.pt`，
`torch.jit.script` 的精确线性权重，float64，与 numpy 侧 `apply_model`
逐位对拍）。期 2（W3-b）要把训练出的模型**作为集体变量注入模拟核心**
——metadynamics 偏置它、restraint 拉它、`colvar.tsv` 记录它。

注入受阻于一个真实的架构事实（issue-dev-plan #9 行原文）：`CVIR` 是
表达式字符串驱动的（`expression` 是 Lepton 可编译的物理），而
TorchScript 模型不是表达式。但 W1-b 已经打开了 **kind-driven** 的口子：
`RMSDForce` / `CustomNonbondedForce`（coordination）/ `PathCV` 三个 CV
的编译由 `kind` 而非 `expression` 驱动（`kernel/openmm.py _compile_cv`
的三个特殊路径），expression 退化为选择符或文档字符串。PathCV 先例
尤其关键：它是 **CustomCVForce over 内层 RMSDForce d1..dP** 的组合——
内层力的"能量"就是 CV 值（RMSDForce 的能量是一个距离，不是物理势能），
外层 metadynamics 表格再把它当变量消费。

## 决策

**新增 CV kind `"TorchScriptCV"`（CVIR 层）+ 两个内核路径（openmm 适配
器 TorchForce 组合 / fake kernel torch 确定性求值），满足双轨纪律
（settled decision #5）。**

### 1. CVIR 扩展（port 层，spec 形状）

```python
CVIR(
    kind="TorchScriptCV",
    expression="<模型描述/来源指纹（文档用途，不编译）>",
    model_path="model.pt",           # convert 产出的 TorchScript 文件
    inputs=[                          # 特征接线：期 1 featurizer 的 spec 文法
        {"type": "distance", "grp1_idx": "10,11", "grp2_idx": "40"},
        {"type": "coordination", "grp1_idx": "10,11", "grp2_idx": "50,51",
         "r0": 0.4},
    ],
    output_index=0,                   # 模型 (k,) 输出中哪一个是 CV 值
    periodic=False,                   # 角度类模型输出可声明周期
    label="mlcv",
)
```

`inputs` 刻意复用 **期 1 featurizer 的 feature spec 文法**（同一套
`type + 键` 词汇，注册表背书）——训练时吃什么特征，注入时就接什么特征，
配置不经过第二次翻译。`inputs` 同时是 fake 路径与 openmm 线性快速路径
（见 §3）的求值依据。

### 2. openmm 适配器：TorchForce 作为 CustomCVForce 的内层变量

PathCV 先例的直接推广：内层 `openmm-torch` 的
`TorchForce(torchscript_module)`，其模型的"能量"**定义为 CV 值**（与
RMSDForce 的"能量是一个距离"同构——CustomCVForce 的变量本来就是内层
力的势能读数，无量纲/自然单位均可）；外层照旧是消费该变量的
CustomCVTableForce（metadynamics）或 restraint 表达式。嵌组合
（表格 CustomCVForce 包 TorchForce）与 PathCV 已验证的
CustomCVForce-inside-CustomCVForce 同一机制。

**包装模型**（期 2 的 convert 升级产出）内部完成 positions→特征→模型
三段：特征层用 torch 原生算子实现 featurizer 原语（选择索引、COM、
Kabsch 对齐、配位开关函数和——全部可微、可 script），线性/非线性模型
随后。期 1 的 convert 产出（forward 吃特征向量）是它的退化子集，期 2
升级为 forward 吃 positions 的完整包装，**权重逐位继承**。

### 3. 分层落地：先线性快速路径，再通用 TorchScript 路径

- **切片 A（零新依赖）**：线性模型（期 1 的全部家当）**expression 化**
  ——CustomCVForce over 特征内层 CV（distance/coordination 等已有的
  编译路径，PathCV 的 d1..dP 形态）+ Lepton 线性表达式
  `w1*d1 + w2*d2 + ... + b`。TICA/logistic 模型直接可注入，不引入
  torch；这是第一个端到端闭环（采样→featurize→train→注入→再采样）。
- **切片 B（本 ADR 主体）**：`TorchScriptCV` 通用非线性路径，torch +
  openmm-torch 进生产依赖。

### 4. fake kernel：torch eval-mode 确定性求值

镜像 PathCV/RMSDForce 的特殊路径：fake 收到 kind ==
`"TorchScriptCV"` 的 bias/CV 时，用 `torch.no_grad()` + `module.eval()`
（关 dropout/batchnorm 训练态）跑同一包装模型，float64→float32→
float64 边界转换与 openmm 路径一致，保证双轨 bit 对拍可 pin。切片 A
的线性路径则完全不需要 torch（表达式求值器足够）。

### 5. 共享接口契约（与 W2-d ML/MM 的 TorchForce 单位契约对齐）

两条轨道都消费 TorchForce，必须共享同一份边界契约，避免两个 seam
各自漂移：

- **输入**：全体系 positions，nm，float32，shape (N, 3)（openmm Context
  的原生精度；float64↔float32 转换只发生在 kernel 边界，包装模型内部
  以 float64 累积特征几何）。
- **输出**：float32 标量（能量语义）或 (k,) 向量（CV 语义）；
  **能量单位 kJ/mol 只在"把模型当力用"的 ML/MM 拼写里成立**——CV 拼写
  下包装模型的输出是模型自然单位的 CV 值（无量纲 TICA 分量 / 概率），
  物理量纲由外层包装 bias（表格、谐波）赋予，与其它每个 CV kind 一致。
- torch/openmm-torch 的 pin 进 pixi.toml 按决策 #10 纪律管理（升级 =
  显式事件：重验私有 API 门 + 重录金样）。

### 6. source-scan 扩展

工作纪律要求"新增 seam 时同步扩展扫描保证"：

- torch 的 import 只允许出现在 `kernel/` 内（openmm.py 的 TorchForce
  编译段 + fake.py 的求值段，或共用 `kernel/_torchcv.py` helper）与
  `mlcv/torch_export.py`（convert 工具）；新增一条 scan 断言。
- `openmm_torch`（`import openmm.torch`）与 openmm 同等待遇：生产适配
  器专属，fake/replay 不得触碰。

## 风险与对策

- **torch pin**：生产路径引入重依赖（决策 #10 纪律：pin + 升级事件 +
  金样重录）；切片 A 保证无 torch 也能跑线性闭环。
- **模型确定性**：eval-mode + no-grad + CPU 归约可确定；CUDA 归约非
  确定——CI 与金样在 fake/CPU 路径，CUDA 只跑统计层（金样现有分层）。
- **力正确性 vs autodiff**：TorchForce 对模型自动微分，力 = -dE/dx 的
  正确性依赖包装模型对 positions 可微（特征层全部用可微 torch 原语）。
  验证形态镜像 W1-b：fake/openmm 双轨值对拍 + 有限差分力检查
  （`energy_forces()` 的力 vs 数值微分，解析几何小体系）。
- **TorchScript 跨版本**：`.pt` 由 pinned torch 的 convert 产出；注入时
  校验 `torch.__version__` 与记录的训练版本一致，不一致即
  `UpstreamVersionError` 风格的显式拒绝（复用 openmm_privates 的门风格）。

## 否决的替代方案

- **PLUMED 后端**（`py-plumed` 接 NN CV）：绕过 KernelPort seam、三适配
  器纪律（fake/replay 无法确定性复现），与 issue #11 路径 A 被拒同理。
- **模型权重硬编码 Lepton 长表达式**：仅覆盖线性、表达式随特征数膨胀
  而脆弱——保留为切片 A 的受控拼写，不作为通用方案。
- **把序列化模型塞进 `CVIR.expression` 字符串**：违背 W1-b 已定的
  kind-driven 决策，且把二进制负担压给一个本应是人类可读物理的字段。

## 后果

- 正面：双轨纪律成立（fake torch 求值 ↔ openmm TorchForce 值对拍）；
  与 ML/MM 共享 TorchForce 底座与单位契约，port 扩展审阅批次
  （issue-dev-plan §四：TorchCV + ml_region + softcore 集中管理）只需
  做一次；featurizer spec 文法一份两用。
- 负面（已认领）：torch 进生产依赖（pin 纪律认领）；fake kernel 跑
  TorchCV 需要 torch（无 torch 环境下该 CV 的测试 skip，注册与校验仍
  可用——与期 1 convert 的 torch 门同款分层）；迭代采样-训练闭环的
  工程量集中在包装模型的 torch 特征层（一次性投入，双轨道复用）。

## 重开条件

openmm-torch 与 openmm 版本组合出现不可用的 pin 冲突；或社区出现维护
良好的「表达式级 NN CV 编译器」（能逐模型确定性编译进 Lepton/内层力，
满足双轨且免 torch）。
