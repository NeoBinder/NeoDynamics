# RBFE（相对结合自由能，λ 窗口）

> 需求：[issue #8](https://github.com/NeoBinder/NeoDynamics/issues/8) ·
> 选型与编排：[ADR-0003](../adr/0003-rbfe-technology-selection.md) ·
> [ADR-0007](../adr/0007-rbfe-lambda-window-orchestration.md) ·
> 姊妹文档：[boresch.md](boresch.md)（标准锚 restraint）

## 原理简述

对突变体×底物系列做定量亲和力/选择性排序：alchemical 扰动把一个
配体/残基"关掉"，λ 窗口从 0（物理态）走到 1（解耦态），相邻窗口的
du 样本经 BAR/MBAR 恢复自由能差，全程配体被
[Boresch 锚](boresch.md) restrain 住。扰动层用 openmmtools `alchemy`
（prepare 边界依赖，独立 `rbfe` pixi env，不进默认依赖）；窗口 λ 经
`KernelSpec.global_parameters` 下置；BAR/MBAR 自实现（numpy-only，
pymbar 仅 gated 对拍）。hybrid topology 以 vendor perses 派生代码
（MIT + attribution）构建。

## 使用

### 单窗 plan（fake kernel 可跑的最小示例）

```yaml
method: rbfe
steps: 50000
temperature: 298
seed: 2026
integrator: {dt: 0.002, friction_coeff: 1.0}
input_files:
  complex: /work_dir/min/last.pdbx
  system: /work_dir/sys_prep/htb/system.xml
output:
  output_dir: /work_dir/rbfe/win_00   # one directory per window
  report_interval: 500                # du.tsv row stride
alchemical:
  lambda_values: {lambda_alchemical: 0.0}   # THIS window's λ
  ladder:                                    # every window, in order
    - {lambda_alchemical: 0.0}
    - {lambda_alchemical: 0.5}
    - {lambda_alchemical: 1.0}
  mock_bias: {grp1_idx: "0", grp2_idx: "1", # fake-kernel test potential
              k_kj_mol_nm2: 50.0, r0_nm: 0.3}
```

要点：`lambda_values` 是**本窗**的 λ（必须取自 `ladder` 的某一项，
`plan.py` collect-all 校验）；`ladder` 是全部窗口的 λ 顺序表。
openmm 档上 `mock_bias` 让位于真实 alchemical `lambda_values`。

### run_ladder（整条阶梯）

```python
from neomd.rbfe import run_ladder
outcome = run_ladder(plan)   # N windows, ladder.json ledger, auto-resume
```

输入一份 `method: "rbfe"` 的 Plan（含 `alchemical.ladder`），输出
`window_00…` N 个窗口目录（每窗一次完整 `drive()`）+ runner 级账本
`ladder.json`。中断窗口自动续跑（manifest epoch 驱动）。

### analysis：BAR / MBAR

```bash
neomd analysis bar  /work_dir/rbfe/win_00 /work_dir/rbfe/win_01
neomd analysis mbar /work_dir/rbfe/win_00 /work_dir/rbfe/win_01 /work_dir/rbfe/win_02
```

BAR 取两窗、MBAR 取整条阶梯，输入就是窗口目录列表（du 带自描述）。

## 产物

- **`du.tsv`**：每窗一条（每行一个观测步、每列一个 ladder 档位，
  kJ/mol；带 λ 参数注释行自描述），resume 时按通用规则截断。
- **`ladder.json`**：runner 级账本——ladder、每窗 λ、du 带末步、
  结果摘要。
- 门：`pixi run -e rbfe test-rbfe`（openmmtools 门控冒烟）。

## 参考文献

- Boresch, Karplus et al., *J. Phys. Chem. B* 107, 9535（2003）——
  [锚 restraint](https://doi.org/10.1021/jp0217839)（见
  [boresch.md](boresch.md)）。
- Bennett, *J. Comput. Phys.* 22, 245（1976）——
  [BAR](https://doi.org/10.1016/0021-9991(76)90078-4)；Shirts &
  Chodera, *J. Chem. Phys.* 129, 124105（2008）——
  [MBAR](https://doi.org/10.1063/1.2978177)。
- LiveCoMS RBFE best practices；OpenFE 工业基准（ChemRxiv 2025）——
  公开基准对拍参照，不作依赖。
- [ADR-0003](../adr/0003-rbfe-technology-selection.md) ·
  [ADR-0007](../adr/0007-rbfe-lambda-window-orchestration.md)。
