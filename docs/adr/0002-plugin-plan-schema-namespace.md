# ADR-0002：插件 plan-schema 命名空间（顶层 `plugins:` 段），而非逐插件白名单键或 meta_set ride-along

- 状态：已确认（2026-09-02）
- 决策者：项目维护者
- 关联：[ADR-0001](0001-neomd2-strangler-migration.md)、drill 记录（examples/gamd_drill/README.md）

## 背景

gamd_drill（`examples/gamd_drill/`）验证了 v2 扩展架的三项机制：注册
（`register("method", ...)`）、发现（`importlib.metadata` 组 `"neomd"` 的
entry-point 扫描）、分发（`drive()` 经 `registry.get("method", ...)` 调
`prepare`）。但 plan YAML 没有插件命名空间：`plan.KNOWN_KEYS` 是封闭白名单，
第三方方法的配置键进不了 Plan——drill 只能把设置塞进
`meta_set["gamd_drill"]`（一个 metadynamics 风味的白名单 mapping 段，
plan.py 只查类型不查键）作为临时 ride-along。后果：(a) 语义挪用（插件
配置伪装成 metadynamics 设置）；(b) plan.py 无法为插件段做键级校验
（未知键被静默忽略，与 collect-all 纪律相反）；(c) 指纹/manifest 虽会
记录（raw 全量进指纹），但读 plan 的人看不出哪些段属于哪个插件。该缺口
是真实 GaMD 插件（#10）与 ML/MM（#12）决策的前置。

## 决策

**Plan 新增唯一顶层保留段 `plugins:`；每个已注册插件拥有且仅拥有
`plugins.<name>.*` 键；插件注册时经 registry 声明其段的 schema——新 rack
kind `"plugin"`，条目为 `PluginSection(required={...}, optional={...})`
（键 -> 描述，形状与 method SCHEMA 的 required/optional 一致），与
`register("method", ...)` 并排出现在插件模块的 import 副作用里。**

细则：

1. **校验（collect-all，与现有错误风格一致）**：`plugins` 本身与每个
   `plugins.<name>` 须为 mapping；未注册的插件名是 `ConfigKeyError`
   （yaml key path `plugins.<name>` + 对已注册插件名的 did-you-mean）；
   已注册段内的未知键是 `ConfigKeyError`（key path
   `plugins.<name>.<key>` + 对该段声明键集的 did-you-mean + known keys
   清单）。≥2 个问题走 `PlanValidationErrors` 聚合，1 个抛具体类型——与
   全文件其余校验同一规则。插件 rack 为空时名称检查**不**降级：写了
   `plugins:` 而无任何插件注册，本身就是"未安装/未加载"的正确诊断
   （与 restraint 类型检查"词表未导入则跳过"不同：插件没有 in-tree
   词表，空 rack 不可能是"还没 import"的中间态）。registry 不可导入时
   跳过（既有防御路径）。
2. **required 键的存在性检查放 `--check-files` 层**（`check_plan_files`），
   与 method-required 键同层——结构层只管"名字与键已知"，存在性是语义
   层的事。值类型/范围校验归插件自己的 `prepare`：核心对段内容保持
   不透明（与 meta_set 今天一致，method SCHEMA 同样不做类型校验）。
3. **指纹**：插件段随 `plan.raw` 全量参与指纹——Plan 是不可变实验快照，
   插件的 `boost_factor` 与 `temperature` 同等地改变指纹。无单独机制，
   也无需（manifest 的 `plan_raw` 记录天然带上插件段）。
4. **配置如何到达 prepare()/方法分发**：prepare 契约不变——
   `prepare(kernel, plan, sink, logger)`，插件自己读
   `getattr(plan, "plugins", None).get(<name>)`。不给 dispatch 加
   kwarg：Plan 本来就作为不可变快照整体下发给 prepare；加参既分叉契约，
   又预设"命名空间 ↔ 方法名"一一对应（不成立：一个发行版可注册
   method + probe + cv 多个条目共享一个命名空间）。
5. **加载时机**：`plugins:` 校验依赖注册表状态，插件必须在 Plan 构造前
   注册。门面负责扫描：`md_run` 与 `compile`（dict 入参分支）在构造
   Plan 前调用 `registry.scan_entry_points()`，`neomd validate` 同样先
   扫描再校验（扫描 = import = 注册，是插件契约本身的副作用，不写盘）。
   库级调用者（直接 `Plan.from_dict` / `drive()`）自行导入或扫描插件——
   与 method 分发的既有纪律相同（driver 只保证 in-tree methods 的
   import-即-注册）。

## 否决的替代方案

### 逐插件顶层键（`gamd_set:`、`opes_set:` …）

- KNOWN_KEYS 封闭白名单是 Plan "指纹永固 + 全量校验"的基础；每个插件
  开一个顶层键意味着核心白名单随插件生态膨胀
- 命名空间冲突（两个发行版都想叫 `boost_set`）没有仲裁机制

### 把 meta_set ride-along 正式化

- 语义挪用制度化：meta_set 是 metadynamics 的方法设置段；把它变成通用
  插件载体等于在 plan 里开一个无 schema 的后门，未知键静默通过，与
  collect-all + did-you-mean 纪律相反
- 插件无法声明键集 → 键级 did-you-mean 与 `neomd validate` 的段级诊断
  都做不了

### 在 method 条目上挂段声明（不加新 kind）

- 预设"有 plan 段的东西必是 method"；probe/cv 插件与"一个命名空间对
  多个 rack 条目"的发行版立刻无处安放

### `plugins.<name>` 深层 schema（类型/范围/嵌套结构）

- 核心开始解释插件配置语义，违背"段内容对核心不透明"；且与 method
  SCHEMA 的浅校验不同构

## 后果

### 正面

- 第三方 method/probe/cv 插件获得受校验、受指纹、带 did-you-mean 的
  配置入口；drill 的 `meta_set` ride-along 与 `gamd_set` 容错读法删除
- KNOWN_KEYS 保持封闭：核心键白名单不随插件增长
- `neomd validate` 对插件段给出与核心键同级的诊断（file:line +
  did-you-mean）

### 负面（已认领）

- Plan 校验对"插件加载状态"有了强依赖：未装插件的 plan 在未装/未扫描
  的进程里一律 unknown plugin——这是快照语义的正确行为，但要求用户
  理解"先装、先扫描"
- 门面每次 run/compile/validate 前多一次 entry-point 扫描（无插件
  安装时是 no-op 读取）
- 插件段内的值校验仍在插件侧——垃圾值要到 prepare 才报错（与 meta_set
  今天相同）

## 重开条件

出现需要核心理解插件配置结构的场景（例如 #12 ML/MM 的 `ml_region` 需要
插件键参与 KernelSpec/内核构建）——届时按 port 扩展批次另立决策，而不是
在本命名空间里加深 schema。
