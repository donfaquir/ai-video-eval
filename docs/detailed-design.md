# video-eval 详细设计

本文档是 design.md 的展开，将架构设计细化为可实现的接口规范。所有类名、方法签名、字段定义均为最终实现依据。

> **修订说明（v7）**：本版解决 v6 评审遗留的 2 个 P0（融合数据路径、VLMJudge 依赖粒度）和 3 个开放决策：
>
> - **A1/v7（P0，融合数据路径）**：`default_weight_of(d)` 在 v6 中无实现路径（融合策略不持有 registry 引用）。v7 为 fuse() 增加第四参数 `default_weights: dict[str, float | None]`，由 Pipeline 从各评估器 `meta.default_weights` 预计算后注入（§2.4 `_build_default_weights`），§5.1 伪代码同步改用 `default_weights.get(d)` 替代未定义的 `default_weight_of()` 函数。
> - **A2/v7（P0，VLMJudge 依赖粒度）**：v6 的 `requires = ["frames", "asr", "ocr"]` 使 VLMJudge 在 ASR/OCR 任一不可用时被 F3 **整体跳过**，违反 design.md §5.5 降级哲学。v7 弱化为 `requires = ["frames"]`（VLM 至少需要视觉帧），ASR/OCR 下沉为**子维度级软依赖**：cross_modal 需 ASR 或 OCR 至少一种，两者均缺则该子维度 skip；hook_strength/marketing_logic/audience_match 为纯视觉评估，ASR/OCR 可用时作为辅助上下文注入 prompt、不可用时仍可评估（§6.1 子维度表与 evaluate() 伪代码同步更新）。
> - **D10（决策，strict_veto_dims + 多槽位）**：strict_veto_dims 条目是**维度名**（槽位名）。多槽位评估器的子维度可单独进入 strict_veto_dims。子维度级软依赖缺失（reason=`missing_dependency`）不在 STRICT_VETO_TRIGGERS 中，不触发否决（§5.3 新增说明）。
> - **D9 probe 生命周期**：probe 实例在 `__exit__()` 后立即释放，不缓存、不复用（§5.3 补充说明）。
> - **交叉引用同步**：§4.2 步骤编号修正、§8.4 plugins 输出更新 VLMJudge requires、§10.1 新增 test_vlm_judge.py 与 test_fusion.py 的 v7 测试点、§11 新增两行差异记录。
>
> ---
>
> **修订说明（v6）**：根据第五轮评审意见（编号 A1-A2 / B1-B5 / C1-C6 / P3；开放决策 **D9 经用户确认采用「启动 probe」方案**）修订。v5 结构与数据模型已收敛，该轮主题是「补齐 D5 的实现手段与交叉引用一致性」：
> - **A1/决策 D9**：strict_veto 依赖完备性引入**启动 probe**——对其 requires 闭包内每个抽取器执行一次 `__enter__()`→`__exit__()` 实测，失败即退出码 2。静态闭包校验拦不住「已注册、设备满足、但依赖未安装」（重依赖按 §2.1 约定在 `__enter__` 内 import，注册/物化阶段一切正常），v5 对此无手段，运行期只会静默降级 → compliance 只审 OCR 文本就以 A/B 放行，口播极限词系统性漏检。§5.3 前置校验表扩为五行，§5.1 的「不应可达」声明自此成立，§8.5 的 `asr(✓)` 判定标准同步（C1）；
> - **A2**：`veto_only_dims` 判定纳入 `default_weights` 兜底（与 raw_weights 同口径）——v5 只查权重表，声明了 `default_weights` 的 veto 维度会被误判为纯否决，兜底声明在有阈值时永不生效，与 §2.1 承诺矛盾；
> - **B1**：`--resource-strategy` 引用断裂（§8.2/§8.3 无此参数）修正为 `--set batch.mode=sequential`；
> - **B2**：§5.3 新增「物化失败」拦截面（strict_veto 维度 import 失败 → 退出码 2，而非 `import_error` 占位静默放行），§1.6 分派规则同步例外；
> - **B3**：`effective_slots` / `slots_for()` 的 `or` 改为显式 `is not None` 判定——空槽位（`dimensions_<video_type>: []`）是合法的「该类型下主动关闭」配置，`or` 会误回退吞掉用户意图；
> - **B4**：D8 上传/预处理缓存容量恒为 1（`video_path` 变化即清空）——LocalBackend 的已归一化帧张量若按 video_path 累积，resident 批量下 N 个视频 = N 份张量常驻，与 §4.1 显存预算冲突；
> - **B5**：补 `ExtractionError` 异常定义（管线级异常，携带抽取器名与异常链）；
> - **C2**：§6.4 注明 aigc_defect 与 clip_features 双模型的原因与合并条件；§4.1/§10.2 显存预算同步补 aigc_defect CLIP-L（≈21GB → ≈22GB）；
> - **C3**：`extraction_failed` 占位结果的 evidence 携带 `extraction_failures` 失败摘要（失败原因不进报告，事后排查只能翻 verbose 日志）；
> - **C4**：抽取器新增 `extractors.<name>.enabled` 开关（与评估器 F1 对称）：禁用后其 provides 字段不可用；required 抽取器被禁用 → warning；strict_veto 依赖的抽取器被禁用 → §5.3 闭包校验拦截（退出码 2）；
> - **C5**：单视频模式显存不足归入 §4.4 步骤 ③ `error: init_failed`（无整批预算概念，C4 显存校验仅 run_batch）；
> - **C6**：退出码 2 改述为「配置/环境前置校验失败」（涵盖 D5/D9 与 C4 新增类别）；
> - **P3 全 5 项**：§4.5 修订注行号过时、run() 步骤 2 的 C4 适用范围、§5.3 batch 口径（effective_slots 并集）、`_backends` 保留键保护（§9.4）、§1.6 失效表两行合并。
>
> v5 及更早修订历史见 git 历史；各节内"修订 XX"标注保留自引入版本的编号。
>
> **关键决策记录**（均可在评审时推翻，推翻需同步修改对应章节）：
> - **D1**：抽取器 optional 失败采用字段级降级而非中断视频（与 design.md §5.5 降级哲学对齐）；
> - **D2**：`judge()` 接收只读 EvalContext（而非仅 frames 或仅 video_path），兼顾 LocalBackend（用 frames）与 APIBackend（上传 video_path）；
> - **D3**：strict_veto 维度被 disabled/未注册 → fail fast（退出码 2），而非强制否决；
> - **D4**：VLM 结果缓存推迟 Phase 2（缓存键需 prompt 模板版本参与，避免过早固化）；
> - **D5**（v5 新增，D3 的语义细化）：strict_veto 的否决只对**执行失败**（`error:*` / `skipped: runtime_unavailable`）触发；**环境不完备**（依赖缺失 / 设备不满足 / 抽取失败）一律在启动前置阶段 fail fast——语义上让用户去改环境，而不是让用户误以为视频违规；
> - **D6**（v5 新增）：`criticality` 作为公开插件契约，社区抽取器可声明 `required`（默认 `optional`，向后兼容）；
> - **D7**（v5 新增）：backend 使用独立配置段 `backends.<name>`，可被多个评估器复用；
> - **D8**（v5 新增）：同一视频的多子维度 VLM 调用复用一次上传/预处理（backend 内按 video_path 缓存，`__enter__`/`__exit__` 生命周期内有效，**容量恒为 1**，B4/v6）；
> - **D9**（v6 新增，经用户确认）：strict_veto 维度依赖完备性以**启动 probe** 实测——对其 requires 闭包内每个抽取器执行一次 `__enter__()`→`__exit__()`，失败即退出码 2（§5.3 校验表第 4 行）。静态校验拦不住「已注册、设备满足、依赖未安装」；probe 仅覆盖抽取器，strict_veto 评估器自身与其 backend 的 `__enter__` 失败仍属「执行失败」（init_failed → 否决），不参与 probe。probe 实例**不复用**（v7 明确）。
> - **D10**（v7 新增）：strict_veto_dims 条目是**维度名**（槽位名），不是评估器名。多槽位评估器的子维度可单独进入 strict_veto_dims。子维度级**软依赖**缺失（如 cross_modal 的 ASR+OCR 均不可用）产出 `skipped: missing_dependency`，该 reason 不在 STRICT_VETO_TRIGGERS 中，不触发否决。这是有意设计：软依赖缺失是评估器内部的降级决策，有别于硬依赖/环境不完备。

---

## 一、Registry 系统

### 1.1 Registry[T] 泛型类

框架核心，泛型、线程安全的映射表：alias → 对象或 Placeholder（字符串路径）。注册时同步登记元数据快照，使 Pipeline 过滤阶段无需物化即可读取类属性。

#### 类签名

```python
T = TypeVar("T")

class Registry(Generic[T]):
    def __init__(
        self,
        name: str,
        base_cls: type[T],
        *,
        allow_override: bool = False,
    ) -> None: ...
```

#### 字段表

| 字段 | 类型 | 可见性 | 说明 |
|------|------|--------|------|
| `name` | `str` | public | Registry 语义名称，用于错误信息 |
| `base_cls` | `type[T]` | public | 基类，注册/物化时做 issubclass 校验 |
| `allow_override` | `bool` | public | 是否允许覆盖已注册条目 |
| `_entries` | `dict[str, T \| Placeholder[T]]` | private | 名称→条目映射，保留注册顺序 |
| `_meta` | `dict[str, PluginMeta]` | private | 名称→元数据快照（注册时写入，无需物化即可读取） |
| `_origins` | `dict[str, str]` | private | 名称→来源标签 |
| `_broken` | `set[str]` | private | 物化失败标记集合，避免重复尝试 |
| `_frozen` | `bool` | private | 冻结标志，True 后禁止注册 |
| `_lock` | `threading.RLock` | private | 注册与物化共用锁，保证线程安全 |

> **修订 A1/B5/B8**：删除 LRU 缓存（物化后 `_entries` 已存真实类，LRU 逐出无意义）；新增 `_meta` 旁路元数据表；新增 `_broken` 失败缓存。

#### 方法签名

| 方法 | 签名 | 说明 |
|------|------|------|
| `register` | `(name: str, *, cls: type[T] \| None = None, origin: str = "explicit") -> Callable \| None` | 注册类。传 cls 立即注册；不传返回装饰器。同步提取类属性写入 `_meta`。冻结后抛 RegistryFrozenError；冲突抛 DuplicateRegistrationError。**持锁执行**（check-then-act 原子性） |
| `get` | `(name: str) -> T` | 按名称获取物化后的类。未找到抛 NameNotFoundError（含智能提示）；命中 Placeholder 且不在 `_broken` 中时触发懒加载；在 `_broken` 中直接抛 MaterializationError（携带首次失败原因，不重试） |
| `get_meta` | `(name: str) -> PluginMeta` | 获取元数据快照，**无需物化**。供 Pipeline 过滤、plugins 命令使用 |
| `has` | `(name: str) -> bool` | 是否已注册（含未物化的 Placeholder） |
| `list` | `() -> list[str]` | 返回所有已注册名称，按注册顺序 |
| `list_meta` | `() -> list[PluginMeta]` | 返回所有元数据快照，按注册顺序 |
| `freeze` | `() -> None` | 冻结，此后 register() 抛 RegistryFrozenError |
| `unfreeze` | `() -> None` | 解冻，约定仅测试代码调用 |
| `origin` | `(name: str) -> str` | 返回来源标签：`builtin` / `entry-point` / `explicit` |
| `is_placeholder` | `(name: str) -> bool` | 条目是否仍为 Placeholder |
| `is_broken` | `(name: str) -> bool` | 条目物化是否曾失败 |

#### 异常类型

| 异常 | 说明 |
|------|------|
| `RegistryError(Exception)` | 所有 Registry 异常基类，携带 registry_name |
| `NameNotFoundError(RegistryError)` | get() 未命中，message 含智能提示（前 10 个可用名称，超出截断） |
| `RegistryFrozenError(RegistryError)` | 冻结后尝试注册 |
| `DuplicateRegistrationError(RegistryError)` | concrete→concrete 冲突 |
| `MaterializationError(RegistryError)` | 物化失败（import 错误或类型校验失败），纳入 RegistryError 家族 |

### 1.2 PluginMeta 数据结构

注册时同步提取的元数据快照，使 Pipeline 过滤、plugins 命令等消费方无需物化即可读取类属性。

```python
@dataclass
class PluginMeta:
    name: str                        # 插件 alias
    version: str                     # 版本号
    device_requirement: str          # "cuda" / "mps" / "gpu" / "any"
    requires: list[str]              # 依赖的 context 字段
    config_schema: dict              # 配置项 schema
    provides: list[str]              # 仅抽取器：产出的 context 字段（评估器/后端/融合为 []）
    criticality: str                 # 仅抽取器："required" / "optional"（其余类型固定 "optional"，不参与判定）
    backend_config_key: str | None   # 仅评估器：声明"选择 VLM 后端"的配置键（其余为 None）
    default_weights: dict[str, float] | float | None  # 仅评估器：融合权重兜底（单槽位取 float，多槽位按子维度取 dict）
    dimension_slots: dict[str, list[str]] | None  # 仅多槽位评估器：video_type → 子维度名列表；单槽位为 None
    origin: str                      # "builtin" / "entry-point" / "explicit"
    is_placeholder: bool             # 是否尚未物化
```

> **修订 A1/B4**：v3 的 `provides` / `backend_config_key` 只加在基类属性表，未同步到 PluginMeta，导致 §4.2/§4.3/§4.5 读不到所需元数据；`dimension_slots` 使占位展开不再依赖 VLMJudge 私有的 `dimensions_{video_type}` 配置键命名。

> **修订 B1/C1（v5）**：新增 `criticality`（抽取器失败语义的声明载体，替代 v4 "框架内硬编码内置名单"的做法，见 §2.3/§6.6）与 `default_weights`（权重兜底的声明位置，§5.1 的 `default_weight(d)` 从此字段读取，而非 v4 语焉不详的 "config_schema.default_weight"）。entry-point 条目在物化前取宽松默认值（`provides=[]`、`criticality="optional"`、`backend_config_key=None`、`default_weights=None`、`dimension_slots=None`），但按 §1.6 的主动物化策略，**这些默认值不会进入任何过滤/编排决策**——它们只在 `plugins` 等快路径命令的列表展示中短暂出现。

### 1.3 EvaluatorInfo 数据结构

Pipeline 内部使用，包装 PluginMeta + 运行时状态。

```python
@dataclass
class EvaluatorInfo:
    meta: PluginMeta                 # 元数据快照
    config: dict                    # 合并后的配置段
    effective_slots: list[str]      # 本次 video_type 下的有效槽位名列表（B3，过滤阶段计算）
    status: str = "pending"         # "pending" / "active" / "skipped" / "error"
    reason: str | None = None       # 跳过/失败原因（机器可读标记，最终写入 EvalResult.reason）
```

> **修订 B3（v5）**：新增 `effective_slots`——v4 规定"config 覆盖只在实例化时生效，静态过滤阶段仍用类属性默认值"，导致**同一份配置下输出的维度集合取决于评估器是否被跳过**（正常执行用 config 覆盖值，被跳过时占位展开用类属性默认值），下游解析与 A/B 对比均会踩坑。
>
> 计算规则（无需物化，config 段本身就是 dict）：
>
> ```
> if meta.dimension_slots is None:            # 单槽位
>     effective_slots = [meta.name]
> else:
>     override_key = f"dimensions_{video_type}"    # 多槽位覆盖键的约定命名
>     override = config.get(override_key)          # B3/v6：显式判定，空列表是合法配置
>     effective_slots = override if override is not None else meta.dimension_slots[video_type]
> ```
>
> **空槽位语义（B3/v6）**：显式配置 `dimensions_<video_type>: []` = 该评估器在此 video_type 下**主动关闭**——F1-F3 照常通过但无槽位可执行：不实例化、不生成占位、不参与融合与退出码判定（等同按 video_type 细粒度的 disabled）。v5 的 `or` 写法会把 `[]` 回退为类属性默认值，吞掉用户意图。
>
> **约定（写入 §9.4 检查清单）**：多槽位评估器的 config 覆盖键名**必须**为 `dimensions_<video_type>`（由 `dimension_slots` 的 key 推导），否则覆盖不生效。运行时 `slots_for()` 与占位展开（§4.4）统一读 `effective_slots`，两路不再漂移。

### 1.4 四个 Registry 实例

| 实例 | 存储类型 | base_cls | 职责 |
|------|---------|----------|------|
| `evaluator_registry` | `BaseEvaluator` | `BaseEvaluator` | 评估器插件 |
| `extractor_registry` | `BaseExtractor` | `BaseExtractor` | 特征抽取器 |
| `backend_registry` | `BaseBackend` | `BaseBackend` | VLM 后端 |
| `fusion_registry` | `BaseFusion` | `BaseFusion` | 融合策略 |

### 1.5 装饰器 API

```python
@register_evaluator("technical_quality")
class TechnicalQualityEvaluator(BaseEvaluator): ...

@register_backend("api")
class APIBackend(BaseBackend): ...

@register_extractor("asr")
class ASRExtractor(BaseExtractor): ...

@register_fusion("weighted_veto")
class WeightedVetoFusion(BaseFusion): ...
```

装饰器行为：接收类 cls → issubclass 校验 → **校验 alias 与 cls.name 一致**（不一致抛 RegistryError；v1 不支持一个类注册多个 alias，避免配置段与 dimension 名歧义）→ 提取类属性构造 PluginMeta → 调用 registry.register()（**锁在 register 内部**，装饰器不额外持锁，避免嵌套加锁歧义）→ 原样返回 cls。

提取清单（B1/C1）：

| 插件类型 | 提取的类属性 |
|---------|------------|
| 全部 | `name` / `version` / `device_requirement` / `requires` / `config_schema` |
| 评估器 | 另含 `backend_config_key` / `dimension_slots` / `default_weights` |
| 抽取器 | 另含 `provides` / `criticality` |
| 后端、融合 | 无额外属性（PluginMeta 中相应字段取默认值） |

### 1.6 懒加载机制

#### Placeholder 类

```python
class Placeholder(Generic[T]):
    def __init__(
        self,
        name: str,
        loader: Callable[[], type[T]],
        *,
        origin: str,
        module_path: str | None = None,
    ) -> None: ...
```

#### 物化逻辑

1. 检查 `_broken` 集合，已在其中则直接抛 MaterializationError（不重试）
2. 获取 RLock → 双重检查（防止并发重复物化）
3. 调用 loader() 触发真实 import
4. 失败 → 加入 `_broken`，抛 MaterializationError（记录原始异常）
5. issubclass 校验，失败抛 MaterializationError
6. 替换 `_entries[name]` 为真实类
7. 更新 `_meta[name]`：is_placeholder=False，并从类属性回填完整元数据（entry-point 条目物化前的宽松默认值此时被真实值覆盖）
8. 释放锁

#### 懒加载适用范围与 entry-point 元数据策略

- **内置插件**（目录扫描）：import 时触发装饰器，cls 已在内存，注册时即为 concrete，PluginMeta 从类属性直接提取——启动性能收益主要来自内置插件的重依赖延迟 import
- **entry-point 插件**：注册时只有 EntryPoint 对象（dist 元数据中**不包含** requires / device_requirement / config_schema 等类属性级信息），物化前 PluginMeta 取宽松默认值：`device_requirement="any"`、`requires=[]`、`provides=[]`、`criticality="optional"`、`config_schema={}`、`backend_config_key=None`、`default_weights=None`、`dimension_slots=None`；首次 `get()` 物化后回填真实值并更新 `_meta`
- **物化时机（B5/A1/C6）**：宽松默认值的真实后果比"运行时兑底"更严重，且**四类插件各有各的失效方式**：

| 插件类型 | 未物化时的宽松默认值 | 会失效的校验/编排 |
|---------|-------------------|-----------------|
| 评估器 | `requires=[]` | 不进入 §4.5 required_keys 并集 → 相关抽取器根本不运行 → 插件拿到 `asr=None` / `ocr=[]` 这类合法空值后大概率**不抛异常而是给出错误低分**（status 仍是 scored） |
| 抽取器 | `provides=[]`、`criticality="optional"` | §4.5 第 4 步按 `provides ∩ required_keys` 选集永不命中 → **永不执行 → 永不物化 → `_meta` 永不回填**（死锁闭环）；连带 §4.2 静态层 available_fields 少算该抽取器的字段 → 依赖它的评估器被 F3 全部误杀；required 抽取器被当 optional 降级，失败后用户拿到一份"全部跳过"的报告而非明确错误 |
| 后端 | `device_requirement="any"` | §4.3 F2 的 backend 设备联动校验失效 → CPU 机器上选一个 gpu-only 的社区 backend 不会被拦，留到 `__enter__` 才炸 |
| 融合 | `config_schema={}` | §7.2 的 fusion 段校验失效 |

  因此：**Pipeline 启动时（run / run_batch 入口）主动物化全部四类 entry-point 插件**（不区分类型——外部插件总量有限，成本可控；CLI 启动提速收益本来就主要来自内置插件的重依赖延迟 import）。物化失败按各自类型的既有语义处理：评估器 → `skipped: import_error` 占位（**例外（B2/v6）**：该维度在 strict_veto_dims 中 → 配置错误，退出码 2，见 §5.3 校验表第 2 行——强否决维度连类都加载不了，是最严重的环境不完备，不得以占位静默放行）；抽取器 → 视 `criticality` 走 §4.5 失败语义；后端 → 若被 `backend_config_key` 选中则为配置错误（退出码 2），未被选中则打 warning 并从注册表视图中标记 broken；融合 → 若被 `fusion.strategy` 选中则为配置错误（退出码 2）。

- **保持快路径的命令**：`--help` / `plugins` / `device info` 不物化（列表展示宽松默认值时，STATUS 列标记 `not loaded`，避免把默认值误读为插件真实声明）；`plugins --detail NAME` 与 `config check` 主动物化目标插件

### 1.7 冲突检测规则

| 当前条目 | 新注册条目 | allow_override | 结果 |
|---------|-----------|----------------|------|
| Placeholder | Concrete | 任意 | **允许升级**，替换并保留 origin |
| Concrete | Concrete | True | 覆盖 |
| Concrete | Concrete | False | **拒绝**，抛 DuplicateRegistrationError |
| Concrete | Placeholder | 任意 | **拒绝**，保留 Concrete，抛 DuplicateRegistrationError |
| Placeholder | Placeholder | 任意 | 允许（保留先注册者） |
| 无 | 任意 | 任意 | 正常注册 |

### 1.8 智能错误提示

`_suggest_similar(name, max_results=3, levenshtein_threshold=3)`：前缀匹配 + 子串匹配 + Levenshtein 距离，综合得分降序取前 3。

NameNotFoundError 消息格式（可用列表截断为前 10 个）：

```
NameNotFoundError: 'tehcincial' not found in evaluator registry.
  Did you mean: 'technical_quality', 'aigc_defect'?
  Available: technical_quality, aigc_defect, compliance, ... (10 of 28, use list() for full list)
```

### 1.9 插件发现机制

#### entry_points（外部插件）

社区插件在 pyproject.toml 声明：

```toml
[project.entry-points."video_eval.evaluators"]
watermark = "video_eval_plugin_watermark:WatermarkEvaluator"

[project.entry-points."video_eval.backends"]
claude = "video_eval_plugin_claude:ClaudeBackend"

[project.entry-points."video_eval.extractors"]
face = "video_eval_plugin_face:FaceExtractor"

[project.entry-points."video_eval.fusions"]
pareto = "video_eval_plugin_pareto:ParetoFusion"
```

四个入口点组名：`video_eval.evaluators` / `video_eval.backends` / `video_eval.extractors` / `video_eval.fusions`。

#### 目录扫描（内置插件）

扫描 `video_eval/evaluators/` 等目录下所有 .py 模块，importlib 触发装饰器注册。origin 标记为 `builtin`。

#### 初始化总顺序

```
1. 创建四个 Registry 实例
2. scan_directory() 四个目录 → builtin 条目（import 触发装饰器，cls 已在内存）
3. discover_entry_points() 四个组 → 外部 Placeholder 条目（同名冲突容错，见下）
4. 不自动 freeze（freeze 仅在 CLI 入口调用，库模式不冻结）
```

**发现阶段容错（B3）**：`discover_entry_points()` 对每个 entry-point 单独 try/except——同名冲突（DuplicateRegistrationError，如第三方包占用内置 alias）、ep.load 失败等均只打 warning 并跳过该条目，不中断扫描，不使 `video-eval --help` 崩溃。被同名内置条目遮蔽的外部插件在 `video-eval plugins` 输出中标记 `shadowed`。

> **修订 B6**：freeze 仅在 CLI 入口生效，库模式不冻结，用户可在 import 后注册自定义评估器。

---

## 二、基类设计

### 2.1 BaseEvaluator 抽象基类

#### 类属性

| 属性 | 类型 | 可选值 | 默认值 | 说明 |
|------|------|--------|--------|------|
| `name` | `str` | — | （必填） | 维度唯一标识 |
| `version` | `str` | — | `"0.1.0"` | 语义化版本号 |
| `device_requirement` | `str` | `"cuda"` / `"mps"` / `"gpu"` / `"any"` | `"any"` | 设备要求。`"gpu"` 表示 cuda 或 mps 均可 |
| `requires` | `list[str]` | — | `[]` | 依赖的 context 字段 |
| `config_schema` | `dict` | — | `{}` | 配置项 Schema |
| `backend_config_key` | `str \| None` | — | `None` | 声明"选择 VLM 后端"的配置键（如 VLMJudge 为 `"backend"`）。F2 会同步校验所选 backend 的 device_requirement。不使用后端的评估器为 None |
| `dimension_slots` | `ClassVar[dict[str, list[str]] \| None]` | — | `None` | 多槽位契约：video_type → 子维度名列表。None = 单槽位（槽位 = [name]）。占位展开（§4.4）与 slots_for() 的数据源；config 可用 `dimensions_<video_type>` 键覆盖（§1.3 `effective_slots`） |
| `default_weights` | `ClassVar[dict[str, float] \| float \| None]` | — | `None` | 融合权重兜底（C1）：单槽位声明 `float`，多槽位声明 `{子维度名: float}`。仅当 config 权重表（`weights_<video_type>`）无该维度时生效；None = 不声明（该维度无权重时 §5.1 打 warning 并按 0 处理） |

#### 多槽位契约方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `slots_for` | `(self, video_type: str) -> list[str]` | 返回该 video_type 下的结果槽位名列表。默认实现：单槽位返回 `[self.name]`；多槽位取 `override = self.config.get(f"dimensions_{video_type}")`，返回 `override if override is not None else self.dimension_slots[video_type]`（B3/v6：显式 `is not None` 判定，空列表是合法的「该类型下主动关闭」配置，`or` 会误回退；与 §1.3 `effective_slots` 同一算法，两路结果必然一致）。未识别的 video_type 抛 ValueError |

> **修订 A1/B3**："多槽位"升级为基类级契约。Pipeline 的占位结果展开读 `EvaluatorInfo.effective_slots`（§1.3，不物化即可算得），不再依赖 VLMJudge 私有的 `dimensions_{video_type}` 配置键命名——**框架只认 `dimension_slots` 类属性 + `dimensions_<video_type>` 这一约定覆盖键名**；插件不得用其他键名做子维度覆盖（v5：v4 只在实例化时应用覆盖，静态过滤阶段用类属性默认值，两路会漂移）。

> **修订 B7**：device_requirement 新增 `"gpu"` 值，表示 CUDA 或 MPS 均可。匹配矩阵：`"gpu"` 在 cuda 或 mps 设备上均满足；`"cuda"` 仅在 cuda 上满足；`"mps"` 仅在 mps 上满足；`"any"` 任何设备均满足。

#### 方法签名

| 方法 | 签名 | 说明 |
|------|------|------|
| `__init__` | `(self, device_manager: DeviceManager, config: dict) -> None` | **仅存引用和配置，不加载重资源** |
| `__enter__` | `(self) -> BaseEvaluator` | 加载模型/分配显存。内部须 try/except 自清理 |
| `__exit__` | `(self, exc_type, exc_val, exc_tb) -> None` | 释放资源，保证幂等 |
| `evaluate` | `(self, context: EvalContext) -> EvalResult \| list[EvalResult]` | 核心评估入口。返回单个或多个结果（多维度评估器如 VLMJudge 返回多个） |
| `check_availability` | `(self) -> tuple[bool, str]` | **实例方法**（非 classmethod），在 `__enter__` 后调用，检查运行时动态条件（如模型文件是否存在、显存是否够）。返回 (可用与否, 人类可读说明)。**说明文本仅写入 verbose 日志**；EvalResult.reason 统一由框架填 `runtime_unavailable`（机器可读枚举），不直接落入 evidence/reasoning |

> **修订 A2/B7/B12**：evaluate() 支持返回 list[EvalResult]（多结果槽位）；check_availability 改为实例方法检查运行时条件；明确 __init__ 不加载重资源、__enter__ 须自清理。

#### __init__ 与 __enter__ 职责边界

- `__init__`：仅存储 device_manager 引用、解析 config、初始化轻量变量。**不加载模型、不分配显存**
- `__enter__`：加载模型、分配显存。**内部必须 try/except，失败时清理已分配资源后 re-raise**
- `__exit__`：释放 `__enter__` 分配的资源。Python 语义保证：`__enter__` 失败时 `__exit__` 不会被调用，因此 `__enter__` 必须自清理
- **重依赖延迟 import 约定**：评估器/抽取器模块的顶层只允许 import 标准库与轻量依赖；torch、whisper、open_clip 等重依赖必须在 `__enter__` 内 import。目录扫描会 import 全部内置模块，顶层重 import 会让 CLI 启动提速收益归零

### 2.2 BaseBackend 抽象基类（VLM 后端能力缝隙）

#### 类属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | （必填） | 后端标识：`"local"` / `"api"` / `"mock"`。**同时是其配置段名**（`backends.<name>`，D7） |
| `device_requirement` | `str` | `"any"` | 同 BaseEvaluator 枚举 |
| `version` | `str` | `"0.1.0"` | 语义化版本号 |
| `requires` | `list[str]` | `[]` | 依赖的 context 字段（占位，保持 PluginMeta 提取一致性） |
| `config_schema` | `dict` | `{}` | 配置项 Schema。声明的是**后端自己的段**（`backends.<name>`）中的键，与调用方评估器的 config_schema 互不干扰（B8/D7） |

#### 方法签名

| 方法 | 签名 | 说明 |
|------|------|------|
| `__init__` | `(self, device_manager: DeviceManager, config: dict) -> None` | 初始化轻量配置。config = `config["backends"][self.name]` 段（D7） |
| `__enter__` | `(self) -> BaseBackend` | 惰性加载模型或建立连接 |
| `__exit__` | `(self, exc_type, exc_val, exc_tb) -> None` | 释放连接/卸载模型；同时清空 D8 的上传/预处理缓存 |
| `judge` | `(self, context: EvalContext, prompt: str) -> VLMResult` | 单次评判：接收只读 EvalContext 与 Prompt 模板文本，返回结构化 VLMResult。语义见下 |

#### judge() 契约（P3-10 拆分）

- **入参语义（决策 D2）**：`context` 为只读 EvalContext，backend 自取所需——LocalBackend 消费 `frames` 避免重复解码，APIBackend 可上传 `video_path`，ASR/OCR 文本按需读取。`prompt` 为已渲染的模板文本（模板选择与渲染由调用方评估器负责）。
- **只读约束（C3）**：**不得修改 context 的任何字段，也不得就地修改 frames 中的图像对象**（resize/归一化必须先拷贝）——同一 context 会被后续评估器与并发的其他子维度调用共享。框架侧提供只读视图（§3.1）做兜底，但对 PIL/ndarray 内部的就地写入无法拦截，靠约定 + code review。
- **解析职责**：在 `judge()` 内部完成 raw_output 解析（正则提取 JSON + 字段校验 + 五级评分映射），失败抛 `VLMOutputParseError`（携带 raw_output 片段）；`VLMResult.raw_output` 始终保留原始文本供调试。
- **上传/预处理复用（C2/决策 D8）**：同一 `video_path` 在一次 `__enter__`/`__exit__` 生命周期内**只上传/预处理一次**——backend 内部按 `video_path` 缓存上传引用（file handle / API file id / 已归一化的帧张量），多子维度调用命中缓存。否则 external 场景 4 个子维度会把同一段视频上传 4 次，`api_concurrency` 只是把 4 次上传并行，配额与带宽消耗不变。缓存实现须**线程安全**（并发调用共用同一 backend 实例）；**容量恒为 1**（B4/v6）——检测到 `video_path` 变化即清空旧条目再写入新值：D8 的动机只是同视频内多子维度复用，单视频级缓存即满足；LocalBackend 的已归一化帧张量若按 video_path 无限累积，resident 批量下 N 个视频 = N 份张量常驻，与 §4.1 显存预算冲突。缓存在 `__exit__` 中释放（API 侧文件如需显式删除，在此完成）。

三种实现：LocalBackend（Qwen3-VL 本地，device_requirement="gpu"）、APIBackend（Gemini/GPT API，"any"）、MockBackend（固定分数，"any"）。backend 的 device_requirement 供 F2 联动校验（§4.3）。

#### 五级评分映射（C3）

Prompt 模板要求 VLM 按 E-VAds 五级（1-5）输出，backend 解析时映射为归一化分数：

```
score = (level - 1) / 4      # 1→0.0, 2→0.25, 3→0.5, 4→0.75, 5→1.0
```

与 §5.2 阈值的对齐关系：thresholds.A=0.75 恰为四级线（五级得 4 即 A），thresholds.B=0.60 位于三、四级之间。修改 thresholds 时需重新标定这一对应关系（写入 §9.4 检查清单：改阈值需评估五级映射对齐）。

**离散化提示（P3-4）**：该映射使 VLM 维度的分数只可能取 `{0, 0.25, 0.5, 0.75, 1}` 五个值。因此若把 VLM 子维度（如 `cross_modal`）加入 `veto_thresholds`，阈值应按**级别**而非连续分数设定——例如 0.3 实际等价于"二级即否决"（0.25 ≤ 0.3），0.5 等价于"三级即否决"。配置 VLM 维度的否决阈值时建议直接取 0.0 / 0.25 / 0.5 / 0.75 之一，避免落在级别之间产生"看起来更严实际相同"的错觉。

#### APIBackend 配置（B8/决策 D7）

API key **只从环境变量读取**，禁止写入 config.yaml：

| 环境变量 | 后端 | 说明 |
|---------|------|------|
| `GEMINI_API_KEY` | api (gemini) | Gemini API 密钥 |
| `OPENAI_API_KEY` | api (openai) | OpenAI API 密钥 |
| `VIDEO_EVAL_VLM_BACKEND` | 任意 | 快速切换后端：`local` / `api` / `mock` |

APIBackend 重试/超时配置写在**后端自己的段** `backends.api`（D7——不再与评估器共用 `evaluators.vlm_judge` 段，见 §7.2/§7.3）：

```yaml
backends:
  api:
    provider: gemini         # gemini / openai
    model: gemini-3-flash    # 段内统一用 model，不再有 api_model
    timeout: 30              # 单次请求超时秒数
    max_retries: 3           # 最大重试次数
    retry_base: 1.0          # 指数退避基数
```

**失败预算（C6）**：批量模式下，某视频的 VLM 子维度调用连续失败（重试耗尽 + parse_failed）达到 `api_max_failures`（默认 5）次时，该视频剩余子维度不再调用，直接产出 `error: evaluation_failed` 占位——防止故障 API 拖垮整个 batch。该键与 `api_concurrency` 都是**调用编排**参数，归属调用方评估器段（`evaluators.vlm_judge`）而非后端段：并发与失败计数由 VLMJudge 的子维度循环控制，backend 只负责单次调用。VLM 结果缓存推迟 Phase 2（决策 D4，开放决策：缓存键需 config_hash + prompt 模板版本 + video_path + dimension 参与）。

### 2.3 BaseExtractor 抽象基类

#### 类属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | （必填） | 抽取器标识，也是其配置段名（config 的 `extractors.<name>`） |
| `provides` | `list[str]` | （必填） | **产出的 context 字段名列表**。一个抽取器可产出多个字段，如 video_meta 抽取器 `provides = ["video_meta", "frames"]`。**两个抽取器 provides 同一字段 → 注册时拒绝**（DuplicateRegistrationError），除非 registry allow_override=True |
| `requires` | `list[str]` | `[]` | 依赖的 context 字段（可消费先序抽取器的产出，如 ocr requires=["frames"]）。用于 §4.5 拓扑排序与依赖闭包扩展 |
| `criticality` | `str` | `"optional"` | **失败语义声明（B1/决策 D6）**：`"required"` = 失败则中断该视频的管线（其 provides 是几乎全部评估器的基础，如 video_meta 的 frames）；`"optional"` = 失败则字段级降级（依赖评估器 `skipped: extraction_failed`）。默认 optional，社区抽取器无需改动即向后兼容 |
| `device_requirement` | `str` | `"any"` | 同 BaseEvaluator 枚举。F3 只统计设备满足的抽取器 |
| `version` | `str` | `"0.1.0"` | 语义化版本号 |
| `config_schema` | `dict` | `{}` | 配置项 Schema |

#### 方法签名

| 方法 | 签名 | 说明 |
|------|------|------|
| `__init__` | `(self, device_manager: DeviceManager, config: dict) -> None` | 仅存引用 |
| `__enter__` | `(self) -> BaseExtractor` | 加载解码器/模型（重依赖在此 import） |
| `__exit__` | `(self, exc_type, exc_val, exc_tb) -> None` | 释放资源 |
| `extract` | `(self, context: EvalContext) -> dict` | 执行抽取。接收**只读** EvalContext——可消费先序抽取器已产出的字段（如 ocr/clip_features 消费 frames，不重复解码视频，落实 design.md "特征抽取只做一次"的生死线）；返回增量字段 dict（如 `{"video_meta": ..., "frames": ...}`），**键必须 ⊆ provides 且不得与已存在字段重复** |

> **修订 A2/B9**：v3 的 `extract(video_path)` 使依赖帧的抽取器拿不到 video_meta 的产出，只能各自重新解码。v4 改为传 context；`requires` 从死属性变为拓扑排序依据。

> **修订 B1（v5）**：新增 `criticality` 类属性——v4 把 required/optional 的区分放在 §6.6 的"内置抽取器静态表"里，框架只能硬编码一份内置名单，与"一切皆插件"的定位冲突，且社区替换 video_meta（自定义抽帧器）时无法声明 required，其失败会被当 optional 降级，用户最终拿到一份"全部跳过"的报告而非明确错误。

### 2.4 BaseFusion 融合策略基类

| 成员 | 签名 | 说明 |
|------|------|------|
| 类属性 | `name: str`（必填）、`version = "0.1.0"`、`requires = []`、`config_schema = {}` | 策略元信息 |
| `__init__` | `(self, config: dict)` | 接收融合配置。**有意例外**：融合是纯计算，不接收 device_manager（与其他基类的 `(device_manager, config)` 签名不同，为有意设计而非疏漏） |
| `fuse` | `(self, results: dict[str, EvalResult], video_type: str, weights: dict, default_weights: dict[str, float \| None]) -> FusionOutcome` | 聚合多维评分 → 融合结果。`default_weights` 由 Pipeline 从各评估器 meta 预计算注入（见下）。**只做决策，不组装 FinalReport**——video_path、ReportMeta（framework_version/device/backend/...）由 Pipeline 采集并组装（见 §3.7），第三方融合插件无需关心 meta 采集 |

**default_weights 参数（v7 修复 A1）**：融合策略不持有 evaluator_registry 引用，`default_weight_of(d)` 的数据路径需由 Pipeline 在调用 fuse() 前预计算。Pipeline 组装方式：

```python
def _build_default_weights(evaluator_infos: list[EvaluatorInfo]) -> dict[str, float | None]:
    """从各评估器 meta.default_weights 构建维度→兜底权重映射。"""
    dw_map: dict[str, float | None] = {}
    for info in evaluator_infos:
        meta_dw = info.meta.default_weights
        if meta_dw is None:
            for slot in info.effective_slots:
                dw_map.setdefault(slot, None)
        elif isinstance(meta_dw, (int, float)):
            dw_map[info.meta.name] = meta_dw        # single-slot: dimension = evaluator name
        else:
            for dim, w in meta_dw.items():
                dw_map[dim] = w                     # multi-slot: per sub-dimension
            # effective_slots 中有但 default_weights 未声明的子维度 → None
            for slot in info.effective_slots:
                dw_map.setdefault(slot, None)
    return dw_map
```

Pipeline 在 `_fuse()` 前调用此函数，将结果作为第四参数传入。第三方融合插件实现 fuse() 时可直接用 `default_weights.get(d)` 替代 `default_weight_of(d)`。

### 2.5 DeviceManager 完整接口

| 属性 | 类型 | 说明 |
|------|------|------|
| `device` | `torch.device` | 最终选定的设备 |
| `dtype` | `torch.dtype` | `float16`（cuda/mps）/ `float32`（cpu） |
| `device_type` | `str` | `"cuda"` / `"mps"` / `"cpu"` |

| 方法 | 签名 | 说明 |
|------|------|------|
| `__init__` | `(self, preferred: str \| None = None)` | preferred 可强制指定 |
| `is_gpu` | `(self) -> bool` | cuda 或 mps |
| `memory_info` | `(self) -> dict` | `{"free_gb": float, "total_gb": float}`。数据来源：cuda → `torch.cuda.mem_get_info`；mps → 无官方 free 查询，用 psutil 系统内存估算并标注**近似值**；cpu → psutil。can_load_model 在 mps/cpu 上同样按近似值估算 |
| `can_load_model` | `(self, param_count: int) -> bool` | 按 param_count × 2 bytes 估算，预留 30% 余量 |
| `satisfies` | `(self, requirement: str) -> bool` | 检查设备是否满足 device_requirement（处理 "gpu" = cuda-or-mps 逻辑） |

---

## 三、数据模型

所有模型继承 `pydantic.BaseModel`，`model_config = {"arbitrary_types_allowed": True}`。

### 3.1 EvalContext

| 字段 | 类型 | 来源 | 说明 |
|------|------|------|------|
| `video_path` | `str` | CLI/输入 | 视频文件绝对路径 |
| `video_meta` | `VideoMeta` | video_meta 抽取器 | 视频元信息 |
| `frames` | `list[FrameItem]` | video_meta 抽取器 | 抽样帧列表 |
| `asr` | `AsrResult \| None` | asr 抽取器 | 语音转写结果 |
| `ocr` | `list[OcrItem]` | ocr 抽取器 | OCR 识别结果 |
| `clip_features` | `Any \| None` | clip_features 抽取器 | CLIP 视觉特征 tensor |
| `product_info` | `ProductInfo \| None` | CLI/输入 | 商品信息（通用 AIGC 场景可为 None） |
| `video_type` | `str` | CLI/输入 | `"main_image"` / `"external"` / `"general"`（通用 AIGC，无商品信息） |
| `extraction_failures` | `dict[str, str]` | `_run_extractors` | **抽取失败记录（B2）**：字段名 → 失败原因摘要（如 `{"asr": "faster-whisper not installed"}`）。`model_config` 中排除序列化（不作为 context 字段输出），供 `_refilter_after_extraction`、`extraction_failed` 占位结果的 evidence（§4.4，C3/v6）与 verbose 日志消费 |

> **修订 B2（v5）**：新增 `extraction_failures`。v4 的 `_refilter_after_extraction(filtered, context)` 只拿 context，而 `asr = None` 既表示"asr 抽取器未入选"也表示"入选但失败"，`ocr = []` 更无法区分"无文字"/"未运行"/"运行失败"——二次过滤要给出 `skipped: extraction_failed` 就必须能区分**失败**与**为空**，否则无法实现。

**只读视图（C3）**：框架向评估器的 `evaluate()`、抽取器的 `extract()`、backend 的 `judge()` 传入的是 `context.readonly()` 返回的**冻结派生视图**（`model_config = {"frozen": True}` 的同字段模型，字段对象共享不拷贝）：对字段的重新赋值会直接抛 `ValidationError`，使"误写 context 污染后续评估器"这类极难排查的问题在开发阶段就暴露。局限：对 `frames[i].image`（PIL/ndarray）内部的就地写入无法拦截，仍靠 §9.4 的约定与 code review。`merge()` 只能在原始 context 上调用（由 `_run_extractors` 持有）。

**基础字段来源清单**（非抽取器产出，由输入直接组装）：

| 字段 | 来源 |
|------|------|
| `video_path` | CLI --video 参数 |
| `product_info` | CLI --product-title / --selling-points / --product-images 组装 |
| `video_type` | CLI --video-type 参数 |

**EvalContext.merge(feats: dict, declared_provides: list[str]) 语义**：将抽取器返回的 dict 按键写入对应字段。**双向校验**（B9）：键必须是 declared_provides 的子集（防止插件返回未声明字段静默覆盖其他抽取器产出）且是 EvalContext 已知字段名，任一不满足抛 ValueError（提示抽取器 provides 声明与实现不符）。

**VideoMeta**：`resolution: tuple[int, int]`、`duration: float`、`fps: float`、`bitrate: int`、`has_audio: bool`

**FrameItem**：`frame_idx: int`、`timestamp: float`、`image: Any`（PIL Image 或 ndarray）

**AsrResult**：`full_text: str`、`segments: list[dict]`（含 `start` / `end` / `text`）、`language: str`

**OcrItem**：`frame_idx: int`、`timestamp: float`、`text: str`、`bbox: list[float]`

**ProductInfo**：`title: str`、`selling_points: list[str]`、`main_image_paths: list[str]`

### 3.2 EvalResult

| 字段 | 类型 | 说明 |
|------|------|------|
| `dimension` | `str` | 结果槽位名。单维度评估器 = 评估器 name；多维度评估器（如 VLMJudge） = 子维度名（如 `"sellpoint_coverage"`） |
| `evaluator` | `str` | 产出该结果的评估器 name（多槽位评估器的所有子维度共享同一评估器名）。消费者据此反查 meta.evaluator_versions |
| `score` | `float` | 归一化分数 [0.0, 1.0]。skipped/error 时为 0.0 |
| `status` | `str` | `"scored"` / `"skipped"` / `"error"` |
| `reason` | `str \| None` | 跳过/失败原因的机器可读标记：`device_unavailable` / `missing_dependency` / `missing_product_info` / `extraction_failed` / `import_error` / `init_failed` / `runtime_unavailable` / `evaluation_failed` / `parse_failed`。scored 时为 None |
| `evidence` | `Any` | 证据数据，结构由维度自定义 |
| `reasoning` | `str \| None` | 评分推理过程 |
| `suggestion` | `str \| None` | 改进建议 |

> **修订 A2/R2**：VLMJudge.evaluate() 返回 `list[EvalResult]`，每个 EvalResult 的 dimension 为子维度名，直接作为 `dimension_results` 的 key；新增 `reason`（统一承载跳过/失败原因，供融合层 veto 扫描、meta.skipped、退出码判定消费）与 `evaluator`（多槽位 → 评估器映射）字段。

### 3.3 VLMResult

| 字段 | 类型 | 说明 |
|------|------|------|
| `score` | `float` | VLM 给出的分数 [0.0, 1.0] |
| `reasoning` | `str` | 评分理由 |
| `evidence` | `list[EvidenceItem]` | 多模态证据列表 |
| `suggestion` | `str` | 改进建议 |
| `raw_output` | `str` | VLM 原始输出文本（调试用） |

**EvidenceItem**：`modality: str`（`"visual"` / `"audio"` / `"text"`）、`timestamp: float \| None`、`detail: str`

### 3.4 FinalReport

| 字段 | 类型 | 说明 |
|------|------|------|
| `video_path` | `str` | 评估视频路径 |
| `video_type` | `str` | `"main_image"` / `"external"` / `"general"` |
| `overall_score` | `float` | 加权综合分数 [0.0, 1.0] |
| `grade` | `str` | `"A"` / `"B"` / `"C"` / `"REJECT"` |
| `passed` | `bool` | A/B 为 True |
| `veto_reasons` | `list[str]` | 触发一票否决的原因 |
| `dimension_results` | `dict[str, EvalResult]` | 各维度结果，key = EvalResult.dimension |
| `suggestions` | `list[str]` | 汇总建议 |
| `meta` | `ReportMeta` | 报告元信息 |

**ReportMeta**：`framework_version: str`、`device: str`、`backend: str`、`vlm_model: str`（后两者：vlm_judge 未运行——被 disabled 或整体跳过——时固定为 `"n/a"`，避免输出 schema 不稳定）、`evaluator_versions: dict[str, str]`（按评估器名）、`skipped: list[str]`（被跳过的**维度名**列表，原因见对应 `dimension_results[d].reason`）、`config_hash: str`、`timestamp: str`（ISO-8601）

**config_hash 计算范围（C4）**：对**合并后的最终生效配置**（插件默认值 + config.yaml + `--set` 覆盖后的结果，不含设备探测、时间戳等运行时信息）做规范化序列化（dict 键排序）后取 SHA-256 前 8 位。这样 `--set` 改动会反映在 hash 中，"同 hash 同结果"的可复现性承诺成立；原始 config.yaml 字节不参与（同语义不同格式的配置应同 hash）。

### 3.5 ValidationIssue

```python
@dataclass
class ValidationIssue:
    severity: str          # "error" / "warning"
    plugin_name: str
    message: str
    field: str | None = None
```

### 3.6 BatchItem 与 BatchItemResult

```python
@dataclass
class BatchItem:
    video_path: str                        # 视频文件路径
    video_type: str                        # "main_image" / "external" / "general"
    product_info: ProductInfo | None = None

@dataclass
class BatchItemResult:
    item: BatchItem
    report: FinalReport | None = None      # 该视频管线中断时为 None
    error: str | None = None               # 管线中断原因（required 抽取器失败等）
```

### 3.7 FusionOutcome

```python
@dataclass
class FusionOutcome:
    overall_score: float                   # 加权综合分数 [0.0, 1.0]
    grade: str                             # "A" / "B" / "C" / "REJECT"
    passed: bool                           # A/B 为 True
    veto_reasons: list[str]                # 触发一票否决的原因
    suggestions: list[str]                 # 汇总建议列表
```

> **修订 B1**：fuse() 只做决策返回 FusionOutcome；FinalReport 的 video_path 与 ReportMeta 由 Pipeline 组装（融合策略拿不到也不该关心这些信息）。

---

## 四、Pipeline 编排

### 4.1 Pipeline 类

`run` 与 `run_batch` 职责彻底分离，各自资源语义自包含：

| 方法 | 签名 | 说明 |
|------|------|------|
| `__init__` | `(self, config: dict, device_manager: DeviceManager)` | 注入配置与设备。**不做任何资源加载** |
| `run` | `(self, video_path: str, product_info: ProductInfo \| None, video_type: str) -> FinalReport` | 单视频评估：抽取器、评估器逐个 `with` 加载/释放，流程自包含 |
| `run_batch` | `(self, items: list[BatchItem]) -> list[BatchItemResult]` | 批量评估：内部按 `batch.mode` 策略管理资源生命周期，结束后统一释放 |

Pipeline 不实现 `__enter__`/`__exit__`（v2 的 Pipeline context manager 设计已废弃）：run / run_batch 各自管理资源，避免"单视频模式是否预加载"的歧义。

#### run_batch 资源策略

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| `resident`（默认） | 抽取器与全部评估器在循环前加载一次、常驻，循环结束后统一释放 | 显存充足（显存预算：Qwen3-VL 8B ≈16GB + SigLIP ≈2GB + Whisper ≈3GB + aigc_defect CLIP-L ≈1GB，合计 ≈22GB，40GB 单卡可容纳；C2/v6：aigc_defect 自带 CLIP 与 clip_features 的 SigLIP 独立，见 §6.4） |
| `sequential` | 抽取器常驻；评估器逐个加载，处理完一个 chunk 后释放再换下一个（外循环=评估器）。**按 chunk 执行**：`batch.chunk_size`（默认 8）个视频为一批，先完成批内全部抽取（context 缓存于内存，内存上界 = chunk_size 个 EvalContext），再进入评估器外循环 | 小显存机器 |

#### run_batch 执行流程（A4）

```
0. 主动物化全部四类 entry-point 插件（A1/C6，见 §1.6）
1. 启动前置校验（D5，见 §7.2）：strict_veto 维度依赖可满足性、所选 backend / fusion 已注册 → 不满足即退出码 2（整批共享，只判一次）
2. 对每个 item 独立执行 _discover/_filter（F1/F2/F3 按 item 的 video_type 与 product_info 判定）
   per_item_filtered[i] = filter(evaluators, available_fields(item_i))
3. 预加载集合：
   evaluators_to_load = union(per_item_filtered)                    # 并集，而非任一单 item 的结果
   extractors_to_load = topo_sort(closure(union(各 item 的抽取器种子集)))   # P3-8/A2：先取并集，再做依赖闭包扩展与拓扑排序
4. resident：
     enter 全部 extractors_to_load 与 evaluators_to_load
     for item in items:
         context = _run_extractors(item, per_item_filtered[item])     # 复用常驻实例，不重复 enter/exit
         item_filtered = _refilter_after_extraction(per_item_filtered[item], context)  # B7：批量模式同样需二次过滤
         results = _run_evaluators(context, item_filtered)            # 逐项跳过不满足者，产出占位
         outcomes.append(assemble(results, item))
     统一 exit
   sequential：
     for chunk in chunks(items, batch.chunk_size):
         extractors enter → 批内逐 item 抽取并缓存 (context, item_filtered)（含二次过滤）→ extractors exit
         for evaluator in evaluators_to_load（逐个 enter）:
             for item in chunk:
                 evaluator 在 item_filtered 中 → evaluate(item.context)；否则产出占位
             evaluator exit
5. 返回 list[BatchItemResult]
```

要点：
- 预加载集合取**并集**，并集内的评估器对某个不满足其 F2/F3 条件的 item 仍逐项跳过并产出占位结果（与单视频模式一致）；
- 抽取器集合同样取**并集**后再做闭包扩展（P3-8）——否则某个 item 需要而未预加载的抽取器在循环内无实例可用；
- sequential 的评估器外循环中，评估器对某 item 的执行顺序跳过不影响结果正确性（评估器相互独立）；
- 内存上界 = chunk_size × 平均 EvalContext 大小（64 帧 PIL + 特征 tensor）+ 1 个评估器模型，写入 §10.2 验收；
- **chunk_size 的权衡**：越大→评估器加载次数越少、内存占用越高；越小→内存越低、模型反复加载越多。sequential **不是**"每个模型只加载一次"（见 §10.2 B6）。

#### 批量模式的抽取失败与结果组装（B7）

- **外层抽取器 `__enter__` 失败**：resident / sequential 两种模式语义一致——按 `criticality` 定性且**整批只定性一次**，不做每视频重试（"依赖未安装 / 模型下载失败"这类原因重试无意义，只会把失败成本乘以 N）：
  - `criticality="required"` 失败 → 整批无法评估，直接退出码 6（不逐视频记 error，避免输出 N 份相同的失败报告）；
  - `criticality="optional"` 失败 → 该抽取器的 provides 字段在**整批**范围内不可用（写入每个 item 的 `context.extraction_failures`），依赖它的评估器逐视频产出 `skipped: extraction_failed` 占位；
  - sequential 下抽取器按 chunk 重新 enter，但**失败定性沿用首个 chunk 的结论**（同一进程内环境不会变），后续 chunk 直接跳过该抽取器。
- **BatchItemResult 组装**：`assemble()` 成功路径填 `report=FinalReport(...)` 与 `error=None`；**仅当该 item 的 required 抽取器失败或其他导致管线中断的异常**时填 `report=None` 与 `error="<原因摘要>"`。评估维度级的 error **不**置 `BatchItemResult.error`——它们体现在 `report.dimension_results[d].status == "error"` 与退出码 4。

> **修订 A4/R7/R8**：v2 的"Pipeline context manager 常驻"与 design.md"串行加载释放显存"决策冲突，且 run()/run_batch() 资源流程交叠。v3 改为：单视频模式保持串行 enter/exit（与 design.md 一致）；批量模式显式配置驻留策略（resident/sequential），显存预算写入文档。

### 4.2 run() 执行流程

```
1. 物化全部四类 entry-point 插件（A1/C6，见 §1.6）
2. 启动前置校验（D5/D9，见 §5.3 五行表）：strict_veto 维度依赖可满足性（静态闭包 + 启动 probe）、所选 backend / fusion 已注册 → 不满足即退出码 2（C4 显存预算校验仅适用于 run_batch，单视频模式见 §4.6）
3. evaluators: list[EvaluatorInfo] = _discover_evaluators(video_type)   # 扫描注册表元数据（get_meta，不物化），同时算出 effective_slots（B3）
4. available_fields = _compute_available_fields(product_info)          # 双层定义见下
5. filtered: list[EvaluatorInfo] = _filter_evaluators(evaluators, available_fields)  # 三重过滤（先过滤）
6. context = _run_extractors(video_path, filtered)                     # 按需抽取（含依赖闭包扩展、拓扑排序与字段级降级）
7. filtered = _refilter_after_extraction(filtered, context)            # 抽取后二次过滤（A3/B2，见 §4.5）
8. results = _run_evaluators(context, filtered)                        # 串行评估
9. results = _fill_placeholders(results, evaluators, filtered, video_type)  # 占位结果补齐（§4.4，按 effective_slots 展开）
10. default_weights = _build_default_weights(evaluators)               # v7/A1：预计算维度→兜底权重映射（§2.4）
11. outcome = _fuse(results, video_type, default_weights)              # 融合 → FusionOutcome
12. report = _assemble_report(outcome, video_path, video_type, ...)    # Pipeline 组装 FinalReport + meta
13. return report
```

> **修订 B1/B2/R3**：先过滤再收集 requires；`available_fields` 双层定义：
> - **静态层**：设备满足（`device_manager.satisfies`）且未被 `extractors.<name>.enabled: false` 禁用（C4/v6）的已注册抽取器 `provides` 并集；
> - **运行时层**：基础字段按本次实际存在性判定——`video_path`、`video_type` 恒可用；`product_info` 仅当 `product_info is not None` 时可用。
>
> 修正 v2 缺陷：v2 将基础字段"视为恒满足"，导致省略 --product-title 时依赖商品信息的评估器不会被 F3 拦截而在执行期 NoneType 崩溃。

### 4.3 三重过滤

对每个 evaluator：

- **F1 enabled 过滤**：`config["evaluators"][name]["enabled"] == False` → 跳过（静默，不生成占位结果）。**config 中未出现的插件默认 enabled=true**（C5）。特殊校验（B2/D3）：strict_veto_dims 中的维度被 disabled 或未注册 → 配置错误，退出码 2（见 §5.3）
- **F2 device 过滤**：`not device_manager.satisfies(meta.device_requirement)` → 跳过，标记 `skipped: device_unavailable`。**若评估器声明了 `backend_config_key`**（如 VLMJudge）：config 所选 backend 名未注册 → **配置错误（退出码 2）**；已注册则同步校验该 backend 的 device_requirement（如 backend="local" 要求 gpu，CPU 机器上直接跳过）→ 跳过，标记 `skipped: device_unavailable`
- **F3 requires 过滤**：`meta.requires` 中有键不在 `available_fields`（§4.2 双层定义）内 → 跳过，标记 `skipped: missing_dependency`（product_info 缺失时 reason 细化为 `missing_product_info`）

> **修订 B1/R1/R3/R5**：F3 基于"设备可用抽取器的 provides 并集 + 基础字段运行时存在性"（v2 按抽取器 name 匹配 required_keys，`frames` 等字段无对应抽取器，依赖帧的评估器会被全部误杀）；F2 增加 backend 设备联动校验。

### 4.4 单评估器执行序列

| 步骤 | 动作 | 失败后果 |
|------|------|---------|
| ① 懒加载 | `registry.get(name)` | MaterializationError → `skipped: import_error` |
| ② 实例化 | `EvalCls(device_manager, config)` | 异常 → `error: init_failed` |
| ③ __enter__ | 模型加载 | 异常 → `error: init_failed`（__enter__ 自清理） |
| ④ 运行时检查 | `check_availability()`（实例方法） | False → `skipped: runtime_unavailable` |
| ⑤ 评估 | `evaluate(context)` | 异常 → `error: evaluation_failed` |
| ⑥ __exit__ | 释放显存 | 始终执行（evaluate 异常时也执行） |

**批量模式**：`resident` 下步骤 ①-④ 对每个评估器只执行一次（首批视频前），⑤ 逐视频执行，⑥ 在批末统一执行；`sequential` 下 ①-⑥ 对每个评估器执行一次（跨视频循环）。评估器在常驻期间对**单个视频**的执行异常按 ⑤ 规则记 `error: evaluation_failed`，不影响其他视频。

#### 占位结果生成规则（R2）

所有 enabled 但最终未产出 scored 结果的评估器，**必须**生成占位 EvalResult 写入 `results`，否则融合层（veto 扫描、权重分摊）、`meta.skipped`、退出码判定均拿不到数据：

- 单槽位评估器被 F2/F3/①/④ 跳过 → 生成 1 条占位（dimension=评估器名，status="skipped"，reason=对应标记）
- 多槽位评估器（`meta.dimension_slots` 非 None）被整体跳过 → 按 `EvaluatorInfo.effective_slots` 逐维度生成占位（reason 同上）——与运行时 `slots_for()` 同源，既不依赖评估器私有的配置键命名（A1），也不会因"是否被跳过"而产生维度集合漂移（B3）
- 评估器执行中 error → 槽位展开规则同上，status="error"，reason="init_failed" / "evaluation_failed" / "parse_failed"
- 子维度级跳过（如 sellpoint_coverage 因 product_info 缺失）→ 仅该子维度占位，status="skipped"，reason="missing_product_info"（由评估器内部生成，见 §6.1）
- 抽取后二次过滤（A3）产生的跳过 → 展开规则同上，reason="extraction_failed"，**evidence 携带 `context.extraction_failures` 中所依赖字段的失败摘要**（C3/v6：如 `{"asr": "asr: ModuleNotFoundError: No module named 'faster_whisper'"}`——失败原因不进报告，事后排查就只能翻 verbose 日志）

### 4.5 特征抽取器按需运行

```
1. required_keys = union(meta.requires for meta in filtered_evaluators)
2. required_keys -= 基础字段                              # video_path / product_info / video_type
3. extractor_metas = [m for m in extractor_registry.list_meta()
                     if device_manager.satisfies(m.device_requirement)        # 设备可用
                     and config["extractors"].get(m.name, {}).get("enabled", True)]  # 且未被禁用（C4/v6）
4. seed = [m for m in extractor_metas if set(m.provides) & required_keys]  # 直接满足评估器需求者
5. closure = expand(seed)                                 # A2：依赖闭包扩展（见下），至不动点
6. extractors_to_run = topo_sort(closure)                 # 按 requires→provides 建图拓扑排序
   # 依赖：抽取器 A 的 requires 中某字段由抽取器 B provides → B 先执行
   # 出现环 → 配置/插件错误，退出码 2
7. context = EvalContext(video_path=..., product_info=..., video_type=...)
8. for m in extractors_to_run:                            # 串行，拓扑序
9.     try:
10.        with extractor_registry.get(m.name)(device_manager, config) as extractor:
11.            feats = extractor.extract(context.readonly())   # 只读视图（C3），可消费先序产出（如 frames）
12.        context.merge(feats, declared_provides=m.provides)
13.    except Exception as e:                             # 含物化失败 / __enter__ 失败 / extract 失败
14.        if m.criticality == "required":                # B1/D6：读插件声明值，不查框架内置名单
15.            raise ExtractionError(m.name) from e       # 中断该视频（退出码 6）
16.        for field in m.provides:                       # B2：字段级降级，且失败原因可追溯
17.            context.extraction_failures[field] = f"{m.name}: {type(e).__name__}: {e}"
18.        log.warning(f"optional 抽取器 {m.name} 失败，其 provides 字段降级：{m.provides}")
19. filtered = _refilter_after_extraction(filtered, context)   # 抽取后二次过滤（A3/B2）
20. return context
```

#### 依赖闭包扩展 expand()（A2）

v4 的第 4 步只选「provides 命中评估器 requires」的抽取器，**漏掉了只被其他抽取器依赖的中间抽取器**（如 ocr 依赖 `frames`，而 `frames` 由 video_meta 提供，但没有评估器直接 requires video_meta 时 video_meta 不会入选 → 第 5 步拓扑排序发现 `frames` 无提供者 → 直接退出码 2，最小配置不可用）。v5 在拓扑排序前插入闭包扩展：

```python
def expand(seed: list[PluginMeta]) -> list[PluginMeta]:
    selected = {m.name: m for m in seed}
    while True:
        satisfied = 基础字段 | union(m.provides for m in selected.values())
        pending = union(m.requires for m in selected.values()) - satisfied
        if not pending:                      # 不动点
            return list(selected.values())
        for key in pending:
            provider = next((m for m in extractor_metas if key in m.provides), None)
            if provider is None:
                raise ConfigError(f"抽取器依赖字段 {key} 无提供者")   # 退出码 2
            selected[provider.name] = provider
```

- **字段唯一提供者约定**：同一字段由多个抽取器 provides 属注册期冲突（§1.7 已保证名称唯一，但 provides 可重叠）→ `expand` 取第一个匹配者并打 warning；建议在 `config check` 中将 provides 重叠列为 warning 级问题
- `pending` 中的字段若其提供者设备不满足（不在 `extractor_metas` 内）→ 同样落入「无提供者」分支，退出码 2 并提示「该字段需要 GPU 抽取器 X」

#### _refilter_after_extraction（A3/B2）

```python
def _refilter_after_extraction(filtered, context):
    failed = set(context.extraction_failures)      # B2：仅「失败」字段，不含「未产出/为空」
    kept = []
    for info in filtered:
        if set(info.meta.requires) & failed:
            info.status = "skipped"                # 就地回填 EvaluatorInfo（§1.3），
            info.reason = "extraction_failed"      # 供 §4.4 占位生成读取
        else:
            kept.append(info)
    return kept                                    # 返回仍需执行的子集；被降级者留在 evaluators 全集中待占位
```

**ExtractionError（B5/v6）**：`class ExtractionError(Exception)`——required 抽取器失败的载体（伪代码第 15 行），携带抽取器名，原始异常经 `raise ... from e` 链式保留。属**管线级异常**（与 RegistryError 家族平行，不混入插件异常体系）；run/run_batch 捕获后按 §4.6「中断范围」处理，CLI 以退出码 6 退出。

#### 抽取器失败语义（A3/决策 D1/B1）

`criticality` 由抽取器**自己声明**（§2.3 类属性，默认 `"optional"`，见决策 D6），框架不维护内置名单：

| criticality 声明值 | 失败后果 |
|-----------|---------|
| `"required"`（内置 video_meta 声明此值，provides frames+video_meta 为几乎全部评估器的基础） | 中断**该视频**的管线：单视频模式上抛（退出码 6）；批量模式记入 `BatchItemResult.error` 后继续（--fail-fast 则停整批） |
| `"optional"`（asr / ocr / clip_features 等可选依赖；社区抽取器默认值） | **字段级降级**：其 provides 的字段写入 `context.extraction_failures`，进入抽取后二次过滤——已通过 F3 但依赖这些字段的评估器改标 `skipped: extraction_failed` |

> **修订 A2/A3/B1/B2/B9/R1/R6**：v5 补依赖闭包扩展（A2）、`criticality` 改读插件声明值（B1/D6）、失败字段写入 `extraction_failures` 使二次过滤可实现（B2）、extract 传入只读视图（C3）。v4 重写本节——extract 传入 context（可消费先序产出，落实"特征抽取只做一次"）；引入拓扑排序（v3 按注册顺序执行，把正确性交给文件名）；optional 抽取器失败按字段级降级而非中断整条视频（与 design.md §5.5"没装 whisper 时 ASR 相关评估器标记 unavailable，不报错"的降级哲学对齐——Whisper 装不上不应连累纯规则的 technical_quality/compliance）；merge 增加双向校验。批量模式下抽取器常驻（§4.1），单视频伪代码中逐抽取器的 enter/exit 由批量循环外层承担。

### 4.6 异常处理总表

| 发生位置 | 触发条件 | 标记状态 | 对管线影响 |
|---------|---------|---------|------------|
| 过滤阶段 | enabled=False | 不记录（静默跳过） | 无 |
| 过滤阶段 | device 不满足（含 backend 联动） | `skipped: device_unavailable` | 该评估器不执行，生成占位结果 |
| 过滤阶段 | requires 缺失 / product_info 缺失 | `skipped: missing_dependency` / `missing_product_info` | 该评估器不执行，生成占位结果 |
| 懒加载 | import 失败 | `skipped: import_error` | 跳过该评估器，生成占位结果 |
| 实例化 | __init__ 异常 | `error: init_failed` | 跳过该评估器，生成占位结果 |
| 资源就绪 | __enter__ 异常 | `error: init_failed` | 跳过（__enter__ 自清理），生成占位结果 |
| 运行时检查 | check_availability() False | `skipped: runtime_unavailable` | 跳过，生成占位结果 |
| 资源预算（C4） | resident 模式下 `device_manager.can_load_model()` 拒绝加载（预估显存峰值超预算；仅 run_batch） | —（启动前置校验） | **整批前置失败，退出码 2**，提示改 `--set batch.mode=sequential`（B1/v6）或减少 enabled 评估器；不进入运行期 error，也不生成占位结果。单视频模式无整批预算概念——单个评估器/后端加载显存不足由其 `__enter__` 自然失败，归入步骤 ③ `error: init_failed`（C5/v6） |
| 核心评估 | evaluate() 异常 | `error: evaluation_failed` | 记录后继续下一个 |
| required 抽取器 | 任意异常 | —（上抛 `ExtractionError`，B5/v6） | **中断该视频的评估**（batch 默认记录后继续） |
| optional 抽取器 | 任意异常 | 其 provides 字段写入 `extraction_failures` | 依赖字段的评估器 → `skipped: extraction_failed`（二次过滤） |

**抽取器失败的中断范围（R9/A3）**：中断的是**当前视频**的管线，而非整个批量任务——

- 单视频模式（`run` / CLI `eval`）：required 抽取器异常向上抛给 CLI，退出码 6
- 批量模式（`run_batch` / CLI `batch`）：默认记录该视频的 `BatchItemResult(error=...)` 后**继续下一个视频**；`--fail-fast` 时立即停止整个 batch，已完成的报告正常输出

---

## 五、融合决策

### 5.1 WeightedVetoFusion（默认策略）

#### fuse() 流程

`veto_dims = set(veto_thresholds.keys()) | set(strict_veto_dims)`；`STRICT_VETO_TRIGGERS = {"init_failed", "evaluation_failed", "parse_failed", "runtime_unavailable"}`（D5）。

```
1. 否决扫描（仅对 status == "scored" 的结果检查分数；status 检查先于分数检查）
   for name in veto_dims:
     result = results.get(name)
     if result is None: continue
     if result.status != "scored":
       if name in strict_veto_dims and result.reason in STRICT_VETO_TRIGGERS:
         # D5：仅「执行失败」触发否决——模型跑了但没跑成，风险与不合规等价
         rejected = True
         veto_reasons.append(f"{name} 执行失败（{result.reason}），强否决维度无结果")
       elif name in strict_veto_dims:
         # 环境不完备（missing_dependency / device_unavailable / extraction_failed /
         # import_error）已在启动前置校验 fail fast（退出码 2，见 §5.3 五行表——含
         # D9 启动 probe 实测依赖可加载），正常运行时不应可达；本分支仅为插件自定义
         # reason 等意外路径兜底，记 warning 后跳过，不把环境问题误报为「视频违规」
         log.warning(f"强否决维度 {name} 因环境原因未执行（{result.reason}），未被前置校验拦住")
       continue
     threshold = veto_thresholds.get(name, 0.0)
     if result.score <= threshold:
       rejected = True
       veto_reasons.append(f"{name} 得分 {result.score:.2f} 未高于否决阈值 {threshold}")

2. 权重计算（A3：只排除「纯否决」维度，不排除「既有阈值又有权重」的维度）
   scored_dims = [d for d, r in results.items() if r.status == "scored"]
   veto_only_dims = {d for d in veto_dims               # A2/v6：与下方 raw_weights 兜底链同口径
                     if weights.get(d) is None and default_weights.get(d) is None}
   weighted_dims = [d for d in scored_dims if d not in veto_only_dims]
   if rejected:
     overall_score = 0.0
   elif len(weighted_dims) == 0:
     overall_score = 0.0; grade = "REJECT"  # 全部 skipped/error，或仅纯否决维度有分
   else:
     raw_weights = {}
     for d in weighted_dims:
       w = weights.get(d)
       if w is None:
         w = default_weights.get(d)          # C1/v7：权重表 → default_weights 参数兜底（Pipeline 预计算自 evaluator meta）
       if w is None:
         log.warning(f"维度 {d} 无权重（权重表与 default_weights 均未声明），按 0 处理")
         w = 0.0
       raw_weights[d] = w
     total = sum(raw_weights.values())
     if total == 0:
       overall_score = 0.0  # 所有维度权重为 0
     else:
       normalized = {d: w/total for d, w in raw_weights.items()}
       overall_score = sum(normalized[d] * results[d].score for d in weighted_dims)

3. 等级判定
   >= thresholds.A → A, >= thresholds.B → B, >= thresholds.C → C, 否则 REJECT

4. 建议生成
   suggestion_threshold = config["fusion"]["thresholds"]["B"]  # 联动配置，不硬编码
   suggestions = [f"[{d}] {results[d].suggestion}" for d in scored_dims
                  if results[d].score < suggestion_threshold and results[d].suggestion]

5. 返回 FusionOutcome(overall_score, grade, passed, veto_reasons, suggestions)
   # FinalReport 与 ReportMeta 由 Pipeline._assemble_report 组装（B1，见 §3.7）
```

> **修订 A3/A4/C1（v5）**：
>
> - **A3（评分正确性）**：v4 的「加权维度排除全部 `veto_dims`」把 `product_fidelity`、`aigc_defect` 这类**既在 `veto_thresholds` 又在权重表**的维度一并踢出加权（默认配置下 external 类型仅剩 3-4 个维度参与计分，占权重总和不足一半，且用户在 config 里调这些维度的权重完全无效）。v5 改为只排除 `veto_only_dims`（有阈值但**无权重**，如 `compliance`）：纯否决维度不参与加权也不告警（保留 B7 本意），而「阈值 + 权重」双声明的维度既否决也计分——**两者共存是合法配置**（§7.3 默认值就是这么写的）。v6 修正判定口径：`veto_only` = 权重表与 `default_weights` **均**未声明（v5 只查权重表，声明了 `default_weights` 的 veto 维度会被误判为纯否决，其兜底声明在有阈值时永不生效，与 §2.1 承诺矛盾）；
> - **A4/D5（强否决语义收紧）**：strict_veto 的否决只对**执行失败**生效（`STRICT_VETO_TRIGGERS`）。v4 的「skipped 或 error 均直接拒绝」会与 A3 引入的 `extraction_failed` 交叉出环境错误——例如 compliance 依赖 `asr`、本机没装 whisper，则**每个视频都被 REJECT 并标「强否决维度缺失」**，用户会把它读成内容违规而不是环境问题；环境不完备一律在启动前置 fail fast（§5.3 / §7.2，退出码 2），v6 以 **D9 启动 probe** 补全其实现手段——静态闭包校验拦不住「已注册、设备满足、依赖未安装」（faster-whisper 只在 asr 抽取器 `__enter__` 的 import 处暴露），probe 之前该路径实际可达且静默漏检；
> - **C1（权重兜底位置，v7 补数据路径）**：`default_weights.get(d)` 的数据源为 Pipeline 预计算的 `dict[str, float | None]`（§2.4 `_build_default_weights`），底层读评估器类属性 `default_weights`（§2.1）：单槽位评估器声明 `float`（直接当作该维度权重），多槽位评估器声明 `dict[子维度名, float]`（按子维度取，缺子维度键则视为 None）。融合策略不持有 registry 引用，v7 以显式参数注入替代 v6 语焉不详的 `default_weight_of()` 函数。

> **修订 A5/B11/C8/P3-9/R11/B7/B8/B1/C5**：否决仅对 scored 结果检查分数且 status 先于分数；`skip_reason` 改为 `EvalResult.reason`；否决文案修正为"未高于"（`<=` 语义与 compliance 阈值 0.0 必须否决 score=0.0 的需求一致，不改判断符）；fuse 返回 FusionOutcome；建议阈值联动 thresholds.B；权重和不要求为 1（归一化逻辑兑底）。

### 5.2 等级阈值表（可配置）

| 配置键 | 默认值 | 等级 |
|--------|--------|------|
| `fusion.thresholds.A` | 0.75 | A（优秀，可直接使用） |
| `fusion.thresholds.B` | 0.60 | B（良好，可使用但建议优化） |
| `fusion.thresholds.C` | 0.40 | C（一般，建议修改后再用） |
| 低于 C 或否决 | — | REJECT（不通过） |

### 5.3 strict_veto_dims（强否决维度）

```yaml
fusion:
  strict_veto_dims: [compliance]  # 执行失败（STRICT_VETO_TRIGGERS）时直接拒绝；环境不完备由启动前置校验拦截（下表，D5/D9）
  veto_thresholds:
    compliance: 0.0
    product_fidelity: 0.3
    aigc_defect: 0.3
```

compliance 在**执行失败**时（init_failed / evaluation_failed / parse_failed / runtime_unavailable）不再静默放行，直接 REJECT 并在 veto_reasons 中标明"强否决维度无结果"——模型跑了但没跑成，与不合规的业务风险等价。

**strict_veto_dims 与多槽位评估器（决策 D10/v7）**：strict_veto_dims 条目是**维度名**（槽位名），不是评估器名。单槽位评估器的维度名=评估器名（如 `compliance`）；多槽位评估器的子维度可单独进入 strict_veto_dims（如 `cross_modal`）。当多槽位评估器的某个子维度在 strict_veto_dims 中时：
- 前置校验（下表）检查的是该子维度**所属评估器的 requires 闭包**（而非子维度本身的软依赖）：评估器 requires 不可用 → 所有子维度均无法执行 → 前置拦截。
- 子维度级软依赖缺失（如 cross_modal 的 ASR+OCR 均不可用）产出 `skipped: missing_dependency`，该 reason **不在** STRICT_VETO_TRIGGERS 中 → **不触发否决**。这是有意设计：软依赖缺失是评估器内部的降级决策（可用时更精准、不可用时跳过该子维度），而非环境不完备（有别于硬依赖 frames 缺失的性质）。
- 若业务确需强制 cross_modal 可执行（包括其软依赖 ASR/OCR 必须可用），应同时将 `compliance`（硬依赖 ASR+OCR）加入 strict_veto_dims，或编写自定义前置校验插件。

**环境不完备不走否决，而是 fail fast（A4/C5/决策 D5；实现手段补全于 D9/v6）**：强否决维度因**环境原因**未能执行（依赖缺失 `missing_dependency` / 设备不满足 `device_unavailable` / 抽取失败 `extraction_failed` / import 失败 `import_error`）时，**不得转化为 REJECT** —— 否则本机没装 whisper 就会导致整批视频全部被标为「强否决维度缺失」，用户把环境问题误读为内容违规。正确做法是在**启动前置阶段**就拦住：

| # | 前置校验项 | 校验手段 | 不满足时 |
|---|-----------|---------|----------|
| 1 | strict_veto_dims 中的维度被 `enabled: false`、未注册、或不在本次任何评估器的槽位中（batch 混合 video_type 时口径 = 各 item `effective_slots` 的并集） | 静态（registry + config + effective_slots） | 配置错误，退出码 2 |
| 2 | strict_veto_dims 中的维度物化失败（entry-point 评估器 import 失败，`is_broken`） | 静态（§1.6 主动物化后检查） | 配置错误，退出码 2（B2/v6：强否决维度连类都加载不了，不得以 `import_error` 占位静默放行） |
| 3 | strict_veto_dims 中每个维度**所属评估器**的 `requires`（硬依赖）在当前环境闭包可达（§4.2 静态层 available_fields + §4.5 闭包扩展可覆盖，含设备满足与抽取器未禁用判定；D10/v7：对多槽位评估器取评估器级 requires，不含子维度级软依赖） | 静态 | 配置/环境错误，退出码 2，提示可操作项：装依赖 / 启用被禁用的抽取器 / 把该维度移出 strict_veto_dims |
| 4 | **闭包内每个抽取器的重依赖真实可加载** | **动态（D9 启动 probe）**：对闭包内每个抽取器执行一次 `__enter__()` → `__exit__()` 实测，实例化后立即释放 | 退出码 2，错误文案含探测失败的抽取器名与原始异常（见下） |
| 5 | strict_veto_dims 中维度所选 backend / 设备要求不满足 | 静态（backend 注册 + device_requirement） | 同上（退出码 2） |

五行均在 `config check` 与 run/run_batch 启动前执行（防 `--set` 绕过）；probe 在批量模式下整批只执行一次。

**D9：启动 probe 的必要性（v6）**：静态校验（上表 1/2/3/5）能拦住「未注册 / 被禁用 / 设备不满足 / 闭包断裂」，但拦不住「已注册、设备满足、**依赖未安装**」——抽取器的重依赖按 §2.1 约定在 `__enter__` 内 import，注册与物化阶段一切正常，faster-whisper 缺失只有在实例真正加载时才暴露。若无 probe，该路径运行期的结局是：asr 字段级降级 → compliance `skipped: extraction_failed` → reason 不在 STRICT_VETO_TRIGGERS 中 → **不否决放行**——compliance 实际只审查了 OCR 文本就以 A/B 通过，口播极限词系统性漏检，而这正是强否决要防的业务风险。probe 把判定提前到启动阶段：探测失败即退出码 2，用户拿到「环境坏了，去修环境」而不是一份看起来正常的报告。成本：strict_veto 维度通常 1 个、其闭包内抽取器 ≤4 个，一次性加载-释放为秒级。**边界**：probe 仅覆盖抽取器；strict_veto 维度评估器自身与其 backend 的 `__enter__` 失败仍属「执行失败」（`error: init_failed` → STRICT_VETO_TRIGGERS 否决）——评估器/后端是运行期执行单元，其失败语义已在 §5.1 定义，不与环境探测混同。

**Probe 实例生命周期（决策，v7）**：probe 创建的抽取器实例在 `__exit__()` 后**立即释放，不缓存、不复用**。理由：①probe 阶段无真实 EvalContext（无 video_path），实例内部状态未完成初始化（如 video_meta 未真正解码），复用会引入脏状态；②probe 的唯一目的是验证 `__enter__` 中的 import/模型加载能成功，不是预热；③成本可忽略（strict_veto 维度通常 1 个、闭包≤4 个抽取器，加载-释放为秒级，后续正式执行时再次加载的时间在总延迟中占比极低）。

错误文案示例（§8.5 `config check`）：

```
ERROR  fusion.strict_veto_dims: compliance 依赖字段 asr 的提供者（asr 抽取器）启动探测失败
       faster-whisper 未安装（ModuleNotFoundError: No module named 'faster_whisper'）
       强否决维度必须可执行，否则每个视频都会被错误地归因为违规。
       修复：pip install 'video-eval[asr]'  或  从 fusion.strict_veto_dims 中移除 compliance
```

第 1 行即 v4 的**防绕过校验（B2/决策 D3）**：否则用户把 compliance 配成 disabled 即可静默绕过强否决，与 §5.3 立意直接冲突；第 2 行是 v6 新增的物化失败拦截面（B2）；第 3-4 行是 v5 引入、v6 以 D9 补全实现手段的环境可满足性校验（C5/D5/D9）——第 3 行静态判定闭包与设备，第 4 行动态实测依赖可加载，缺一不可。

---

## 六、内置评估器详细设计

### 6.1 VLMJudge

#### 概述

VLMJudge 是最复杂的内置评估器，使用 VLM（Qwen3-VL / Gemini / GPT）作为裁判，按 E-VAds 的"证据 + 五级评分"格式输出。一个 VLMJudge 实例产出多个子维度结果。

#### 类定义

```python
@register_evaluator("vlm_judge")
class VLMJudge(BaseEvaluator):
    name = "vlm_judge"
    version = "0.1.0"
    device_requirement = "any"       # 评估器本身 any；所选 backend 的设备要求由 F2 联动校验
    backend_config_key = "backend"   # F2 据此校验评估器 config 段（evaluators.vlm_judge）中 backend 键指向的后端设备要求
    requires = ["frames"]            # v7/A2：仅硬依赖 frames（VLM 至少需要视觉输入）；ASR/OCR 为子维度级软依赖（见下）
    dimension_slots = {               # A1：基类级多槽位契约；框架只认此类属性 + `dimensions_<video_type>` 这一约定覆盖键
        "main_image": ["sellpoint_coverage", "cross_modal"],
        "external": ["hook_strength", "marketing_logic", "audience_match", "cross_modal"],
        "general": ["cross_modal"],
    }
    default_weights = {              # C1：按子维度声明权重兜底（config 权重表缺该维度时生效）
        "sellpoint_coverage": 0.20, "cross_modal": 0.15,
        "hook_strength": 0.20, "marketing_logic": 0.15, "audience_match": 0.10,
    }
    config_schema = {
        "backend": {"type": "str", "default": "mock", "required": True},   # B8/D7：只保留后端选择键
        "api_concurrency": {"type": "int", "default": 4},    # 编排参数（属评估器，非后端）
        "api_max_failures": {"type": "int", "default": 5},   # 同上：失败预算
        "dimensions_main_image": {"type": "list", "default": ["sellpoint_coverage", "cross_modal"]},
        "dimensions_external": {"type": "list", "default": ["hook_strength", "marketing_logic", "audience_match", "cross_modal"]},
        "dimensions_general": {"type": "list", "default": ["cross_modal"]},
    }
```

> **修订 B8/D7（v5）**：`model` / `api_provider` / `api_model` / `api_timeout` / `api_max_retries` / `api_retry_base` 全部从本节移到**后端自己的配置段** `backends.local` / `backends.api`（§2.2 / §7.3）。v4 把后端配置挤在 `evaluators.vlm_judge` 内，结果：第三方 backend 的键对 VLMJudge 而言是未知键 → 每次运行都告警；且 `model`（local 模型名）与 `api_model` 语义撞车。`api_concurrency` / `api_max_failures` 是 **VLMJudge 的编排参数**（控制子维度并发与失败预算，与具体后端无关），留在本段。

> **修订 P3-2（v5）——子维度数据源只有一条链**：`dimension_slots` 类属性是**声明**，`dimensions_<video_type>` 是**约定覆盖键**（键名由 `dimension_slots` 的 key 推导，框架不接受其他命名）。两者合并结果在**过滤阶段**就算入 `EvaluatorInfo.effective_slots`（§1.3），`slots_for()` 与占位展开（§4.4）读同一源，不存在 v4 的「静态用类属性、运行时用 config」双轨。

#### 子维度

| 子维度名 | 适用 video_type | 子维度级软依赖 | 缺失时行为 | 说明 |
|---------|----------------|-------------|-----------|------|
| `sellpoint_coverage` | main_image | `product_info` | skip: `missing_product_info` | 商品卖点覆盖情况 |
| `cross_modal` | 两者 / general | `asr` 或 `ocr`（任一可用即可） | 两者均缺失 → skip: `missing_dependency` | 口播/花字与画面一致性 |
| `hook_strength` | external | —（纯视觉） | — | 前 3 秒钩子强度 |
| `marketing_logic` | external | —（纯视觉） | — | 营销逻辑完整性 |
| `audience_match` | external | —（纯视觉） | — | 目标受众匹配度 |

子维度随 video_type 变化（已在 config_schema 声明默认值）：

```yaml
vlm_judge:
  dimensions_main_image: [sellpoint_coverage, cross_modal]
  dimensions_external: [hook_strength, marketing_logic, audience_match, cross_modal]
  dimensions_general: [cross_modal]
```

**子维度级软依赖（v7/A2，R3 扩展）**：

v6 的 `requires = ["frames", "asr", "ocr"]` 使 VLMJudge 在 ASR 或 OCR 任一不可用时被 F3 **整体跳过**（所有子维度丧失），违反 design.md §5.5 降级哲学。v7 将 ASR/OCR 从评估器级硬依赖下沉为**子维度级软依赖**：

- **硬依赖（评估器级 requires）**：仅 `["frames"]`。VLM 至少需要视觉帧作为输入。frames 不可用时 F3 跳过整个 vlm_judge（合理：无视觉输入则无法评估）。
- **软依赖（evaluate() 内按子维度判定）**：
  - `product_info`：仅 sellpoint_coverage 需要，缺失 → 该子维度 skip。
  - `asr` / `ocr`：cross_modal 评估口播/花字与画面的一致性，需要**至少一种文本模态**。ASR+OCR 均不可用（均在 `extraction_failures` 中）→ cross_modal skip: `missing_dependency`；只要任一可用即可评估（prompt 按实际可用模态组装）。
  - hook_strength / marketing_logic / audience_match：**纯视觉评估**，VLM 仅依据视频帧判断。ASR/OCR 可用时作为辅助上下文写入 prompt（提升评分精度），不可用时仍可评估（prompt 只含视觉帧）。

- **Prompt 自适应组装**：evaluate() 中为每个子维度组装 prompt 时，检查 `context.asr`（非 None 且对应字段不在 `extraction_failures`）与 `context.ocr`（非空列表）的可用性，可用则注入对应文本段落，不可用则省略。EvalResult.evidence 中标注实际输入的模态集合（`"input_modalities": ["visual", "asr"]`），使结果可解释。

#### 调用编排

VLMJudge 对每个子维度**独立调用一次** VLM 后端（默认 dimensions 列表下 external 为 4 次）。api 后端时单视频内多子维度可按 `api_concurrency` 并行；local 后端固定串行。理由：
- 每个维度有独立的 Prompt 模板，混在一起会降低输出质量
- 单次调用只评一个维度，VLM 输出更聚焦、更稳定
- 某个维度解析失败不影响其他维度

**并发与配额（C2/决策 D8）**：`api_concurrency` 只能把多子维度的**推理请求**并行化，不得把上传/预处理也重复 N 次——同一 `video_path` 在一次 `__enter__`/`__exit__` 生命周期内只上传/预处理一次（缓存在 backend 内，需线程安全，见 §2.2 judge() 契约）；否则 4 个子维度 = 4 次整片上传，并发只是把配额与带宽消耗并行花掉而不是减少。同时多个子维度共享同一 EvalContext，backend 对 frames 的 resize/归一化等处理**不得就地修改**（C3）。

#### backend 生命周期（A5）

VLMJudge 接管所选 backend 的完整生命周期，与 §4.4 执行序列对齐：

```
__init__:  BackendCls = None（不获取，仅存 config）
__enter__: backend_name = self.config["backend"]            # 评估器 config 段中的 backend 键
           BackendCls = backend_registry.get(backend_name)   # 名称不存在 → NameNotFoundError
           backend_cfg = full_config["backends"][backend_name]        # B8/D7：后端独立段
           self._backend_cm = BackendCls(device_manager, backend_cfg)  # 段不存在时传 schema 默认值
           self._backend_cm.__enter__()                      # 连接/模型加载
           # 获取或进入失败 → error: init_failed（步骤③，与评估器自身 __enter__ 失败同口径）
           # backend 未注册已在 F2/config check 前置拦截（退出码 2）
__exit__:  self._backend_cm.__exit__(...)                    # 关闭 backend（含清空 D8 上传缓存）；幂等，未进入时不调用
```

- Pipeline 向评估器传入的 config 为其自身段（`evaluators.vlm_judge`），但需额外传入 `backends` 段以供上述查找（实现上由 `BaseEvaluator.__init__` 接收的 config 携带 `_backends` 只读引用，具体传参路径见 §7.2）；
- **resident 批量模式**下评估器常驻 → backend 随之常驻（API 连接 / 本地 16GB 权重只加载一次）；
- backend 的 `check_availability` 不单独调用——评估器 ④ 的 check_availability 可顺带检查 backend 状态。

#### Prompt 模板管理

Prompt 模板放在 `video_eval/prompts/` 目录，每个子维度一个 .txt 文件。VLMJudge 在 `__enter__` 时加载模板。模板内含 E-VAds 式的五级评分标准和证据溯源要求。

#### 输出解析与容错（解析职责归 backend）

**解析职责**：backend 在 judge() 内部完成输出解析——调用 API/模型后将 raw_output 解析为结构化 VLMResult（正则提取 JSON + 字段校验）。解析失败时 backend 抛 `VLMOutputParseError`（框架内置异常，携带 raw_output 片段）；VLMResult.raw_output 保留原始文本供调试。

**VLMJudge 转换**：

```
VLMResult.score      → EvalResult.score
VLMResult.reasoning  → EvalResult.reasoning
VLMResult.evidence   → EvalResult.evidence
VLMResult.suggestion → EvalResult.suggestion
dimension            → 子维度名
evaluator            → "vlm_judge"
```

**解析失败策略**：backend 抛 VLMOutputParseError → VLMJudge 捕获，该子维度标记 `error: parse_failed`（reason 承载原因），score=0.0，不重试（不浪费 API 调用）。解析失败不写入 ReportMeta，通过 dimension_results[d].reason 与 verbose 日志可查。

#### evaluate() 返回值

```python
def evaluate(self, context: EvalContext) -> list[EvalResult]:
    dimensions = self.slots_for(context.video_type)   # 基类契约方法（config 覆盖已生效）
    # v7/A2：子维度级软依赖判定
    asr_available = (context.asr is not None
                     and "asr" not in context.extraction_failures)
    ocr_available = (bool(context.ocr)
                     and "ocr" not in context.extraction_failures)
    results = []
    for dim in dimensions:
        # 子维度级依赖检查
        if dim == "sellpoint_coverage" and context.product_info is None:
            results.append(self._placeholder(dim, reason="missing_product_info"))
            continue
        if dim == "cross_modal" and not asr_available and not ocr_available:
            results.append(self._placeholder(dim, reason="missing_dependency"))
            continue
        # Prompt 自适应组装：按可用模态注入文本段落
        prompt = self._build_prompt(dim, context,
                                    include_asr=asr_available,
                                    include_ocr=ocr_available)
        # 决策 D2：传只读 context——LocalBackend 消费 frames，APIBackend 上传 video_path
        vlm_result = self.backend.judge(context, prompt)
        result = self._convert(dim, vlm_result)
        # evidence 标注实际输入模态（可解释性）
        result.evidence["input_modalities"] = (
            ["visual"]
            + (["asr"] if asr_available else [])
            + (["ocr"] if ocr_available else [])
        )
        results.append(result)
    return results
```

### 6.2 Compliance

#### 概述

合规审查评估器，检测极限词、医疗违规词、敏感实体。纯规则，不需要模型。

#### 类定义

```python
@register_evaluator("compliance")
class ComplianceEvaluator(BaseEvaluator):
    name = "compliance"
    version = "0.1.0"
    device_requirement = "any"
    requires = ["asr", "ocr"]
    default_weights = None           # C1：compliance 是纯否决维度（只在 veto_thresholds 中，不入权重表），不参与加权（A3）
    config_schema = {
        "limit_words": {"type": "list", "default": [], "required": False},
        "medical_words": {"type": "list", "default": [], "required": False},
        "banned_entities": {"type": "list", "default": [], "required": False},
    }
```

#### 评分语义

- 发现任何极限词或医疗违规词 → score = 0.0，status = "scored"，evidence 列出命中词和来源
- 未发现任何违规 → score = 1.0，status = "scored"
- **不做部分评分**（1 个违规词和 10 个都是 0.0），因为合规是二值判断——要么合规要么不合规

#### evidence 结构

```json
{
  "violations": [
    {"type": "limit_word", "word": "全网最低", "source": "asr", "timestamp": "3.2-4.5s"},
    {"type": "medical_word", "word": "根治", "source": "ocr", "timestamp": "7.0s"}
  ],
  "total_violations": 2,
  "text_length": 245
}
```

### 6.3 TechnicalQuality

纯规则评估器，device_requirement="any"，requires=["frames"]。评估分辨率、模糊度、闪烁、黑边。evidence 为各子项分数和具体数值。

### 6.4 AIGCDefect

device_requirement="gpu"，requires=["frames"]。v0 用 CLIP zero-shot 检测"扭曲/变形/不自然"类瑕疵。evidence 为被标记的帧列表（frame_idx + timestamp + defect_prob）。

**双模型说明（C2/v6）**：本评估器自带的 CLIP（`openai/clip-vit-large-patch14`，§7.3 `evaluators.aigc_defect.model`）与 clip_features 抽取器的 SigLIP（`ViT-SO400M-14-SigLIP-384`）是**两份独立模型**——zero-shot 瑕疵分类与图文相似度匹配对模型能力的要求不同，v0 保持独立加载（resident 模式显存预算按两份计算，§4.1）。若实现阶段验证 SigLIP 可兼顾 zero-shot 检测，可改为消费 `clip_features` 并删除 `model` 键以省一份加载（开放项，暂不并入 v1 范围）。

### 6.5 ProductFidelity

device_requirement="gpu"，requires=["frames", "clip_features", "product_info"]。用 SigLIP 计算视频帧与商家主图的图文相似度。requires 包含 clip_features（需要 CLIP 抽取器先运行）与 product_info（无商品信息时 F3 直接跳过，reason="missing_product_info"，而非执行期 NoneType 崩溃）。evidence 为每个卖点的最佳匹配帧和相似度。

### 6.6 内置抽取器总表（C1）

| 抽取器 | provides | requires | device | criticality（声明值） | 失败语义 | 配置段（extractors.*） |
|--------|----------|----------|--------|------------------|---------|----------------------|
| `video_meta` | `[video_meta, frames]` | `[]` | any | **`"required"`** | 中断该视频（退出码 6） | 共享键 fps/max_frames（见 §7.2 配置传参路径） |
| `asr` | `[asr]` | `[]`（自行从视频文件提取音频） | any | `"optional"` | 字段级降级：依赖 asr 的评估器 → `skipped: extraction_failed` | model_size / language / beam_size |
| `ocr` | `[ocr]` | `[frames]` | any | `"optional"` | 字段级降级：依赖 ocr 的评估器 → `skipped: extraction_failed` | confidence |
| `clip_features` | `[clip_features]` | `[frames]` | gpu | `"optional"` | 字段级降级（同时被 F2 设备过滤排除，字段不可用） | model_name |

说明：
- `criticality` 是**抽取器自己的类属性声明**（B1/D6），上表列的是内置抽取器各自的声明值，而非框架维护的静态名单；社区抽取器不写则默认 `"optional"`，替换 video_meta 的自定义抽帧器可自行声明 `"required"`；
- 拓扑序：video_meta → {ocr, clip_features}；asr 与 ocr / clip_features 之间**无先后约束**，拓扑序内任意位置均可（实现上串行执行，P3-1）；
- clip_features 在无 GPU 设备上被 §4.5 第 3 步排除，字段不可用 → product_fidelity 被 F3 跳过（而非执行期崩溃）；
- 每个抽取器均可通过 `extractors.<name>.enabled: false` 显式禁用（C4/v6，与评估器 F1 对称）：禁用后其 provides 字段不可用（§4.2 静态层排除），依赖评估器被 F3 跳过；strict_veto 维度依赖的抽取器被禁用 → §5.3 第 3 行闭包校验拦截（退出码 2）；required 抽取器（video_meta）被禁用仅 warning（§7.2，用户可能确有只跑无帧依赖评估器的边缘场景）。

---

## 七、配置系统

### 7.1 三层优先级

| 优先级 | 来源 | 说明 |
|--------|------|------|
| 1（最高） | CLI 参数 | `--set evaluators.compliance.limit_words=最,第一` |
| 2 | 配置文件 | `config.yaml` |
| 3（最低） | 插件默认值 | config_schema 中声明的 default |

CLI 参数格式：`--set <dotted.key>=<value>`。**键名与配置文件层级完全一致**（根键 evaluators 为复数）。类型转换规则（schema 驱动优先）：
- 若目标键在对应插件 config_schema 中已声明 → 按 schema type 强制转换（转换失败报配置错误，退出码 2）
- 未声明（自定义键）→ 启发式推断：int → float → bool（true/false）→ 逗号分割 list → str
- 示例：`--set evaluators.aigc_defect.defect_threshold=0.8` → float 0.8
- 示例：`--set evaluators.compliance.limit_words=最,第一,独家` → list ["最", "第一", "独家"]

> **修订 C3/P3-2/R13**：统一 CLI 参数风格为 `--set key=value`，键名与配置根一致（evaluators 复数，v2 单复数不一致导致 merge 失配）；类型转换 schema 驱动优先。

### 7.2 ConfigLoader

| 方法 | 签名 | 说明 |
|------|------|------|
| `__init__` | `(self, config_path: str \| None = None)` | |
| `load` | `(self) -> dict` | 读取 config.yaml + 合并插件默认值 |
| `merge_cli_overrides` | `(self, config: dict, cli_args: list[str]) -> dict` | 解析 `--set` 参数并覆盖 |
| `validate` | `(self, config: dict) -> list[ValidationIssue]` | 校验配置 |

配置校验规则：
- 引用了未注册的评估器（config 中 enabled=true 但 registry 中不存在）→ **error**，退出码 2（fail fast）
- config.yaml 不存在时 → 使用默认配置 + 打印 warning
- required 字段缺失 → error
- 评估器配置段中的**未知键**（不在 config_schema 中）→ warning（不阻断，容忍插件版本差异）
- **`backends.<name>` 段中的未知键**（不在该 backend 的 config_schema 中）→ warning（B8/D7：每个 backend 只按自己的 schema 校验自己的段，第三方 backend 的键不再对 VLMJudge 报未知键）；`backends` 下未注册的后端名段 → warning（可能是未安装的插件预留配置），仅当它被 `evaluators.*.backend` 选中时才升为 error
- **strict_veto_dims 中的维度被 disabled 或未注册**（B2/决策 D3）→ error，退出码 2（强否决维度不可被静默绕过，见 §5.3）
- **strict_veto_dims 中维度的依赖在当前环境不可满足**（C5/D5/D9）→ error，退出码 2：§5.3 五行校验（disabled/未注册 → 物化失败 → 闭包可达 → **启动 probe** → backend 设备）任一不满足即拦截；错误文案需给出可操作修复项（装依赖 / 启用抽取器 / 把该维度从 strict_veto_dims 移出），见 §5.3 示例
- **backend_config_key 指向的 backend 名未注册**（C7）→ error，退出码 2（fail fast，不留到评估器 `__enter__` 才失败）
- **resident 模式下预估显存峰值超预算**（C4，仅 run_batch；单视频模式无整批预算概念，运行期单评估器加载失败归入 `init_failed`，C5/v6）→ error，退出码 2，提示改 `--set batch.mode=sequential`（B1/v6）或减少 enabled 评估器（见 §4.6）
- **required 抽取器被显式禁用**（`extractors.<name>.enabled: false` 且 criticality="required"，如 video_meta）→ warning（C4/v6，不阻断：提示 frames 等基础字段将不可用、几乎所有评估器会被 F3 跳过；用户可能确有只跑无帧依赖评估器的边缘场景）
- **已启用但无权重的非 veto 维度**（当前 video_type 权重表无此键、评估器 `default_weights` 亦未声明，且不在 veto_thresholds）→ warning（C1/C5："装了等于没装"——该维度会 scored 但不影响 overall_score，提醒补权重或在插件里声明 `default_weights`）

**配置传参路径（C2/B8）**——各组件 `__init__(device_manager, config)` 收到的 config 段：

| 组件 | 收到的段 | 说明 |
|------|---------|------|
| 评估器 | `config["evaluators"][<name>]`（并携带 `_backends` = `config["backends"]` 只读引用） | 插件默认值已合并；`_backends` 仅供声明了 `backend_config_key` 的评估器取所选后端的段（§6.1），不参与未知键校验 |
| 抽取器 | **`extractors` 顶层共享键 + `extractors.<name>` 子段合并（子段优先覆盖）** | `fps` / `max_frames` 为全体抽取器共享的顶层键（video_meta 抽帧与 ocr 检测共用采样参数）；私有配置放各自子段 |
| backend | **`config["backends"][<backend name>]`**（D7） | 与调用方评估器的段彻底分离，同一 backend 可被多个评估器复用；段不存在时传其 config_schema 默认值 |
| 融合策略 | `config["fusion"]` 段 | （纯计算，不接收 device_manager） |

> **修订 C6/B2/C7/C5/C2/B8/C4（v5）**：引用未注册评估器 = 配置错误，fail fast；未知键仅 warning，且 backend 段按自己的 schema 校验（B8）；strict_veto 新增**环境可满足性**前置校验（C5/D5）；resident 显存预算纳入前置校验（C4）；抽取器传参 = 共享顶层键 + 子段覆盖；权重兜底源改为 `default_weights`（C1）。

### 7.3 完整配置 Schema

```yaml
device:
  preferred: auto

extractors:
  fps: 1
  max_frames: 64
  asr:
    enabled: true              # C4/v6：可显式禁用（禁用后 asr 字段不可用，依赖评估器 F3 跳过；
                               # strict_veto 维度依赖的抽取器被禁用 → 退出码 2，§5.3 第 3 行）
    model_size: large-v3
    language: auto
    beam_size: 5
  ocr:
    confidence: 0.5
  clip_features:                  # 配置段名 = 抽取器 name（与 provides 字段名一致）
    model_name: ViT-SO400M-14-SigLIP-384

backends:                         # B8/决策 D7：后端独立配置段，段名 = backend name，可被多个评估器复用
  local:
    model: Qwen/Qwen3-VL-8B-Instruct
  api:
    provider: gemini              # gemini / openai
    model: gemini-3-flash         # 段内统一用 model，不再有 api_model
    timeout: 30
    max_retries: 3
    retry_base: 1.0
  mock: {}                        # 无配置项

evaluators:
  technical_quality:
    enabled: true
  aigc_defect:
    enabled: true
    model: openai/clip-vit-large-patch14
    defect_threshold: 0.6
  product_fidelity:
    enabled: true
  compliance:
    enabled: true
    limit_words: ["最", "第一", "国家级", "顶级", "极品", "全网最低", "独家"]
    medical_words: ["治疗", "疗效", "药到病除", "根治"]
    banned_entities: []
  vlm_judge:
    enabled: true
    backend: mock           # local / api / mock（后端自身参数在 backends.<name> 段，D7）
    api_concurrency: 4      # 多子维度并发度（仅 api 类后端生效，编排参数）
    api_max_failures: 5     # 单视频失败预算，超出则剩余子维度直接占位
    dimensions_main_image: [sellpoint_coverage, cross_modal]
    dimensions_external: [hook_strength, marketing_logic, audience_match, cross_modal]
    dimensions_general: [cross_modal]

fusion:
  strategy: weighted_veto
  strict_veto_dims: [compliance]
  thresholds:
    A: 0.75
    B: 0.60
    C: 0.40
  veto_thresholds:
    compliance: 0.0               # 纯否决：不在任何 weights_* 中 → veto_only，不参与加权
    product_fidelity: 0.3         # A3：既有阈值又在权重表 → 既参与否决也参与加权（合法配置）
    aigc_defect: 0.3              # 同上
  weights_main_image:
    technical_quality: 0.15
    aigc_defect: 0.15
    product_fidelity: 0.20
    sellpoint_coverage: 0.20
    cross_modal: 0.10
    # compliance: veto（不在权重表中，单独用 veto_thresholds 控制）
  weights_external:
    technical_quality: 0.10
    aigc_defect: 0.10
    product_fidelity: 0.10
    hook_strength: 0.20
    marketing_logic: 0.15
    audience_match: 0.10
    cross_modal: 0.05
    # compliance: veto
  weights_general:               # 通用 AIGC 场景（无商品信息）
    technical_quality: 0.35
    aigc_defect: 0.35
    cross_modal: 0.30
  # 权重和不要求为 1，融合层会自动归一化

batch:
  mode: resident                 # resident / sequential（见 §4.1 run_batch 资源策略）
  chunk_size: 8                  # 仅 sequential 生效：每批缓存多少个视频的 context（内存上界依据，§10.2）

output:
  format: json
  include_meta: true
  include_evidence: true
  pretty: true
```

> **修订 B5/B8/A3（v5）**：新增独立 `backends:` 段（D7，`model` / `api_*` 从 `evaluators.vlm_judge` 迁出）；补齐 v4 遗漏的三个键 `api_concurrency` / `api_max_failures` / `batch.chunk_size`（本节自称"完整"且是 config.yaml.example 的来源，不应缺项）；在 `veto_thresholds` 下注明「阈值 + 权重双声明是合法配置」（A3）。

> **修订 C4（v6）**：抽取器段新增 `enabled` 开关（示例见 asr；所有抽取器同义，默认 true，框架消费键不传给抽取器 `__init__`）。

> **修订 A2/P3-4/P3-6/P3-9/R1/R7**：移除不存在的 cta_detection 权重；dimensions 按 video_type 分叉并新增 general；compliance 标记为 veto 不出现在权重表中；权重和不要求为 1；YAML 中文词加引号；抽取器配置段名与 name 一致（clip_features）；新增 batch.mode 与 weights_general。

---

## 八、CLI 设计

### 8.1 命令总览

| 命令 | 用途 |
|------|------|
| `video-eval eval` | 评估单个视频 |
| `video-eval batch` | 批量评估 |
| `video-eval plugins` | 列出/查看插件 |
| `video-eval config check` | 校验配置 |
| `video-eval device info` | 查看设备 |

### 8.2 eval 命令

```
video-eval eval [OPTIONS]

必需参数：
  --video PATH                  视频文件路径
  --video-type [main_image|external|general]   # general = 通用 AIGC 场景，无商品信息

可选参数：
  --product-title TEXT           商品标题（通用 AIGC 场景可省略，相关维度自动跳过）
  --selling-points TEXT...       核心卖点列表
  --product-images PATH...       商家主图路径
  --config PATH                  配置文件路径（默认 ./config.yaml）
  --output PATH                  输出路径（默认 stdout）。单视频固定输出单个 JSON 文档，无 --output-format 选项（jsonl 仅对 batch 有意义）
  --set KEY=VALUE ...            临时覆盖配置项（可多个，如 --set evaluators.compliance.limit_words=最,第一）
  --device [auto|cuda|mps|cpu]
  --no-meta                      不输出 meta 字段。优先级：CLI 显式参数 > config 的 output.include_meta（--no-* 总是生效）
  --no-evidence                  不输出证据。优先级同上（> output.include_evidence）
  -v, --verbose
```

> **修订 P3-3**：--product-title 改为可选。省略时 ProductInfo=None，依赖 product_info 的评估器（product_fidelity, sellpoint_coverage）按 F3 自动跳过。

### 8.3 batch 命令

```
video-eval batch [OPTIONS]

必需参数：
  --input PATH                   视频目录或 manifest.csv 路径（二选一）

可选参数：
  --manifest PATH                CSV 文件路径（与 --input 目录二选一）
  --config PATH
  --output PATH                  输出目录
  --output-format [json|jsonl]   json=每视频一文件，jsonl=单文件
  --device [auto|cuda|mps|cpu]
  -v, --verbose
  --fail-fast                    单视频管线中断（**required** 抽取器失败）时停止整个 batch（默认关：记录失败视频后继续）
                                 # P3-9：optional 抽取器失败只做字段级降级，不触发 --fail-fast
```

> **修订 C4/R9**：--input 和 --manifest 二选一；--input 为目录时扫描 .mp4 文件，商品信息缺失的维度自动跳过；抽取器失败默认只中断**该视频**（记录 error 后继续下一个），--fail-fast 才停止整个 batch；评估维度 error 从不中断 batch（逐视频记录）。

### 8.4 plugins 命令

```
video-eval plugins [OPTIONS]

可选参数：
  --detail NAME
  --available-only
  --type [evaluator|backend|extractor|fusion]

输出格式（REQUIRES 列为该插件声明的依赖字段）：

  NAME               VERSION  DEVICE  REQUIRES           STATUS
  technical_quality  0.1.0    any     [frames]           available
  vlm_judge          0.1.0    any     [frames]           available
  aigc_defect        0.1.0    gpu     [frames]           skipped: device gpu unavailable

`--type extractor` 时第 4 列展示 **PROVIDES**（产出字段，供评估器 requires 对查）：

  NAME           VERSION  DEVICE  PROVIDES            STATUS
  video_meta     0.1.0    any     [video_meta,frames] available
  asr            0.1.0    any     [asr]               available
  ocr            0.1.0    any     [ocr]               available
  clip_features  0.1.0    gpu     [clip_features]     skipped: device gpu unavailable
```

被内置同名条目遮蔽的外部插件标记 `shadowed`（§1.9 B3）。

### 8.5 config check / device info

```
video-eval config check [--config PATH]
video-eval device info
```

config check 输出示例：

```
  ✅ device: mps (Apple M2 Max)
  ✅ extractors: video_meta(✓), asr(✓ probed), ocr(✓), clip_features(✓)
  ✅ evaluators:
     - technical_quality: enabled, config valid
     - compliance: enabled, config valid
     - vlm_judge: enabled, backend=api, config valid
  ⚠️  warnings:
     - aigc_defect requires gpu, will be skipped on mps
     - vlm_judge: unknown config key "foo"
     - backends.claude: section present but backend not registered (plugin not installed?)
     - new_dim: enabled but no weight and no default_weights (will not affect overall_score)
  ❌ errors: none
```

extractors 行的 ✓ = 静态校验通过（注册 + 设备 + 闭包可达）；`probed` = 该抽取器位于某个 strict_veto 维度的依赖闭包内，已通过 **D9 启动 probe**（`__enter__`/`__exit__` 实测，§5.3 校验表第 4 行）——config check 执行与 run 相同的 probe，保证「check 通过 = run 能跑」。

error 触发示例（任一出现 → config check 退出码 2，fail fast）：

- vlm_judge: backend "claude" not registered（C7：backend_config_key 联动校验）
- strict_veto dim "compliance" disabled（B2：强否决维度不可静默绕过）
- strict_veto dim "compliance" 依赖字段 asr 在当前环境不可得（C5/D5：闭包不可达，完整文案见 §5.3）
- strict_veto dim "compliance" 的依赖抽取器 asr 启动探测失败：ModuleNotFoundError: No module named 'faster_whisper'（D9 probe，完整文案见 §5.3）
- resident 模式预估显存峰值 22GB 超出可用 12GB（C4：提示改 --set batch.mode=sequential）

device info 输出示例：

```
  Device:      mps (Apple M2 Max)
  dtype:       float16
  Memory:      28.5 GB free / 64.0 GB total
  PyTorch:     2.4.0

  Available backends:
    local:  ready (Qwen3-VL 8B ~16GB, can load)
    api:    ready (GEMINI_API_KEY set)
    mock:   ready

  Registry status:
    evaluators: 5 registered (3 available)
    backends:   3 registered
    extractors: 4 registered
```

### 8.6 退出码

| 退出码 | 含义 |
|--------|------|
| 0 | 评估完成，grade 为 A 或 B，且无 error 维度 |
| 1 | 评估完成，grade 为 C，且无 error 维度 |
| 2 | 配置/环境前置校验失败（引用未注册评估器、必填项缺失、--set 类型转换失败、strict_veto 依赖不可满足（含 D9 probe 失败）、resident 显存预算超限等），前置短路 |
| 3 | 评估完成，grade 为 REJECT |
| 4 | 评估完成，但存在 error 维度（无论 grade 为何） |
| 5 | 无任何维度产出 scored 结果，且无 error 维度（典型：设备不满足或依赖缺失导致全部评估器 skipped；有 error 时归 4）。**仍输出完整报告**（占位结果 + reason），不静默失败 |
| 6 | 管线中断（抽取器失败；batch 模式下 --fail-fast 触发时同样为 6） |

**优先级规则（R10/B6）**：2 在评估前判定（前置短路，不参与下述比较链）。5 在**融合阶段**判定——scored 维度数 = 0 且 error 维度数 = 0 时取 5：此时 §5.1 仍输出 grade=REJECT，但退出码取 5 而非 3（v3 优先级链不含 5，REJECT→3 会掩盖"根本没评"这一事实）。其余运行后按 **6 > 5 > 4 > 3 > 1 > 0** 取最高优先级——例如 grade=A 但某维度 error → 4；REJECT 且有 error → 4。batch 模式退出码 = 所有视频退出码的最严重值；**排序链不含 2，是因为 2 是整批共享的前置短路**（配置/环境不合法时一个视频都不会跑，直接以 2 退出，P3-3）。业务系统门禁建议：0-1 放行，3-6 拦截。

> **修订 C5/R10**：REJECT=3、error=4 独立编码，并定义多条件并存时的优先级（v2 未定义 grade=A + error 维度并存时取何值）。

### 8.7 实现规范

- 框架：click
- 进度显示：tqdm（stderr）
- 退出码：如上表

---

## 九、插件开发 API

### 9.1 评估器插件示例

```python
from video_eval.core.base import BaseEvaluator
from video_eval.core.registry import register_evaluator
from video_eval.core.schemas import EvalContext, EvalResult

@register_evaluator("watermark_detect")
class WatermarkEvaluator(BaseEvaluator):
    name = "watermark_detect"
    version = "0.1.0"
    device_requirement = "any"
    requires = ["frames"]            # frames 由 video_meta 抽取器产出（provides = ["video_meta", "frames"]）
    config_schema = {
        "sensitivity": {
            "type": "float",
            "default": 0.8,
            "required": False,
            "description": "检测灵敏度阈值"
        }
    }

    def __init__(self, device_manager, config):
        self.device_manager = device_manager
        self.sensitivity = config.get("sensitivity", 0.8)

    def __enter__(self):
        # 加载模型（如有）；失败时 try/except 清理后 re-raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def check_availability(self) -> tuple[bool, str]:
        return True, ""

    def evaluate(self, context: EvalContext) -> EvalResult:
        frames = context.frames
        # ... 检测逻辑 ...
        return EvalResult(
            dimension="watermark_detect",
            score=0.95,
            status="scored",
            evidence="No watermark detected in 64 frames",
        )
```

### 9.2 VLM 后端插件示例

```python
from video_eval.core.base import BaseBackend
from video_eval.core.registry import register_backend
from video_eval.core.schemas import EvalContext, VLMResult

@register_backend("claude")
class ClaudeBackend(BaseBackend):
    name = "claude"                  # 同时是其配置段名：backends.claude（D7）
    version = "0.1.0"                # P3-5：与 §9.4 检查清单对齐
    device_requirement = "any"
    requires = []                    # 不依赖 context 字段（纯 API 后端）
    config_schema = {                # 声明的是 backends.claude 段内的键，与调用方评估器互不干扰（B8）
        "model": {"type": "str", "default": "claude-3-opus"},
        "timeout": {"type": "int", "default": 30},
    }

    def __init__(self, device_manager, config):
        # config 就是 config["backends"]["claude"] 段，不再与评估器共享（B8/D7）
        self.device_manager = device_manager
        self.model = config.get("model", "claude-3-opus")
        self.api_key = None  # 从环境变量读取
        self._uploaded: dict[str, str] = {}   # D8：video_path → 已上传引用，需线程安全

    def __enter__(self):
        import os
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._uploaded.clear()               # D8：生命周期结束即失效

    def judge(self, context: EvalContext, prompt: str) -> VLMResult:
        # 决策 D2：签名为 (context, prompt)；context 只读（C3），不得修改其字段
        # D8：同一 video_path 在本生命周期内只上传一次，多子维度复用
        ref = self._uploaded.get(context.video_path) or self._upload(context.video_path)
        # 调用 Claude API，并在本方法内完成解析（解析职责归 backend，§2.2）
        ...
        return VLMResult(score=0.75, reasoning="...", evidence=[], suggestion="...", raw_output="...")
```

> **修订 B4/P3-5（v5）**：`judge` 签名从 v3 的三参数版 `(video_path, prompt, context)` 修正为 §2.2 定稿的 `(context, prompt)`（决策 D2）——本示例是第三方开发者直接复制的模板，不得与基类契约冲突；同时补齐 `version` / `requires` / `config_schema` 声明，以及独立配置段（D7）与上传复用缓存（D8）。

### 9.3 发布插件包

```toml
[project]
name = "video-eval-plugin-watermark"
version = "0.1.0"
dependencies = ["video-eval>=0.1.0"]

[project.entry-points."video_eval.evaluators"]
watermark = "video_eval_plugin_watermark:WatermarkEvaluator"
```

### 9.4 插件开发检查清单

- [ ] 继承正确基类
- [ ] @register_* 装饰器注册，alias 与 cls.name 一致（v1 强制）
- [ ] 声明 name / version / device_requirement / requires / config_schema
- [ ] 抽取器额外声明 provides（extract 返回 dict 的键 ⊆ provides）
- [ ] 抽取器额外声明 `criticality`（`"required"` / `"optional"`，不写则默认 optional；声明 required 意味着失败会中断该视频，P3-6/B1）
- [ ] 抽取器 requires 只声明其他抽取器 provides 的字段或基础字段（拓扑排序与依赖闭包的依据；消费先序产出，不重复解码视频）
- [ ] 模块顶层不 import 重依赖（torch/whisper/open_clip 等放 __enter__ 内）
- [ ] __init__ 仅存引用，不加载重资源
- [ ] __enter__ 加载重资源，内部 try/except 自清理
- [ ] __exit__ 释放资源，保证幂等
- [ ] check_availability 检查运行时动态条件
- [ ] evaluate() 不修改 EvalContext（只读语义；Pipeline 传入的是冻结派生视图，就地修改会抛异常，C3）
- [ ] backend 的 `judge()` 不得修改 context（包括对 frames 做 resize/归一化等就地操作；多子维度并发下会竞争，P3-6/C2/C3）
- [ ] backend 对同一 video_path 的上传/预处理在一个生命周期内只做一次，缓存需线程安全、容量恒为 1（video_path 变化即清空，B4/v6），`__exit__` 清空（D8）
- [ ] 声明了 `backend_config_key` 的评估器不得读写或覆盖注入 config 中的 `_backends` 保留键（§6.1/§7.2——框架注入的后端段只读引用，写入会污染其他评估器的后端配置）
- [ ] evaluate() 返回 EvalResult 或 list[EvalResult]；未产出 scored 的槽位生成占位结果（含 reason）
- [ ] 多槽位评估器：子维度级依赖在 evaluate() 内处理并生成占位结果
- [ ] config_schema 声明所有配置项及默认值
- [ ] 多槽位评估器声明 dimension_slots 类属性（占位结果展开的数据源，§2.1）；如需支持 config 覆盖，覆盖键名**必须**为 `dimensions_<video_type>`（§1.3 effective_slots 的唯一约定，B3）
- [ ] 需参与加权的维度声明 `default_weights`（单槽位 float / 多槽位 dict；不声明且用户未配权重时该维度不影响 overall_score，C1）
- [ ] 修改 §5.2 等级阈值时重新评估五级评分映射对齐（§2.2）
- [ ] API key 从环境变量读取，不写入 config
- [ ] pyproject.toml 声明 entry_points
- [ ] 基本测试（至少用 MockBackend 跑通）

---

## 十、测试策略与性能验收

### 10.1 测试清单（承接 design.md Phase 1）

| 测试文件 | 覆盖点 |
|---------|--------|
| `test_registry.py` | 注册/装饰器/alias==name 校验/冲突表全组合（含 Concrete→Placeholder 拒绝）/freeze/_broken 失败缓存/智能提示 |
| `test_pipeline.py` | MockBackend 端到端；F2/F3 全分支（含 product_info 缺失、frames 字段映射）；占位结果生成与多槽位展开（含 `effective_slots` 在「执行」与「跳过」两路的一致性，B3，及空槽位=主动关闭，v6）；strict_veto（执行失败走否决 / 环境不完备走前置退出码 2 两路，D5/D9：probe 成功放行、probe 失败退出码 2、未装依赖端到端不产生「看似正常」报告；**多槽位子维度在 strict_veto_dims 中 + 软依赖缺失不触发否决，D10/v7**）；batch 两种模式；**run_batch 并集预加载 + per-item 跳过 + batch 下二次过滤**（P3-7/B7） |
| `test_device.py` | satisfies 匹配矩阵（cuda/mps/gpu/any × 设备）；can_load_model 拒绝时的前置退出码 2（C4） |
| `test_fusion.py` | 否决优先级、全 skipped、权重归一化与缺项告警；**veto_only vs 阈值+权重双声明维度的加权差异**（A3 回归；v6 起 veto_only 判定含 default_weights 口径，A2 回归）；`strict_veto × extraction_failed` 交叉用例（A4 回归防护，P3-7）；default_weights 兑底链（C1）；**fuse() 第四参数 default_weights 注入正确性（v7/A1）** |
| `test_cli.py` | 退出码全表映射（含 5 的判定与优先级链、batch 最严重值、前置 2 不参与排序）、--set 类型转换端到端、--no-meta/--no-evidence 覆盖优先级 |
| `test_extractors.py` | provides 拓扑排序（环检测、requires 无提供者报错）、**依赖闭包扩展**（仅有评估器需 ocr 时 video_meta 仍被纳入，A2/P3-7）、**required 抽取器失败中断该视频**（P3-7/B1，上抛 ExtractionError，B5/v6）、optional 抽取器字段级降级与二次过滤（含 `extraction_failures` 区分失败与空值，B2；占位 evidence 携带失败摘要，C3/v6）、merge 双向校验、provides 冲突注册拒绝、`enabled: false` 禁用后字段不可用（C4/v6） |
| `test_vlm_judge.py` | **子维度级软依赖（v7/A2）**：ASR 不可用时 hook_strength/marketing_logic/audience_match 仍可评（prompt 无 ASR 段）；ASR+OCR 均不可用时 cross_modal skip（missing_dependency）；product_info 缺失时 sellpoint_coverage skip；evidence.input_modalities 标注正确；Prompt 自适应组装（有/无 ASR/OCR 各分支） |
| `test_backends.py` | 五级评分映射（含离散化取值集，P3-4）、输出解析容错（VLMOutputParseError）、重试与失败预算（api_max_failures）、**多子维度只上传一次**（D8）、MockBackend 契约 |
| `test_config.py` | 三层合并、--set 类型转换（schema 驱动与启发式）、未知键 warning（含 `backends.<name>` 段按自己 schema 校验，B8）、未注册评估器 error、strict_veto/backend 联动前置校验（含环境可满足性与 probe 触发，C5/D9；required 抽取器被禁用 warning，C4/v6） |
| `test_registry_entrypoints.py` | entry-point 四类插件主动物化（A1/C6）：物化后 `_meta` 回填真实值、物化失败按类型分派（评估器占位 / 抽取器按 criticality / 选中的 backend·fusion 退出码 2）、快路径命令不物化且 STATUS 标 `not loaded` |

### 10.2 性能验收标准

- `video-eval plugins` / `video-eval --help` 冷启动 < 1s（不含首次模型下载）
- 批量模式（resident）下 N 个视频的模型加载次数 = 1（抽取器与评估器各加载一次）
- sequential 模式下评估器加载次数 = **评估器数 × chunk 数**（= 评估器数 × ⌈N / batch.chunk_size⌉，B6）——**不是「每个模型只加载一次」**；chunk_size 越大加载次数越少、内存峰值越高（§4.1）
- 单视频端到端延迟基线（60s 视频、默认配置；回归对比参考，非硬性门槛）：mock 后端 < 30s；api 后端 < 120s（external 4 次 VLM 调用，受网络支配）；local 后端 < 300s（Qwen3-VL 8B，A100 / M3 Max 级硬件）
- sequential 批量内存上界 = `batch.chunk_size` × 平均 EvalContext 大小 + 1 个评估器模型（§4.1；默认 chunk_size=8 下实测峰值 RSS < 8GB）
- resident 模式显存峰值 ≈ 22GB（Qwen3-VL 16GB + SigLIP 2GB + Whisper 3GB + aigc_defect CLIP-L ≈1GB，§4.1 预算表，C2/v6；超出时 can_load_model 在**启动前置阶段**拒绝并以退出码 2 提示改 `--set batch.mode=sequential`，C4/§4.6）

---

## 十一、与 design.md 的差异与待同步项

本版与 design.md 存在以下差异，实现时**以本文档为准**，下列 design.md 章节待同步：

| design.md 位置 | 旧内容 | 本文档定稿 |
|---------------|--------|-----------|
| §5.4 插件生命周期第 7 步 | 管线跑完后 freeze | 初始化后不自动 freeze；freeze 仅 CLI 入口调用（§1.9） |
| §8.1 输出示例 meta 字段名 | video_eval_version / evaluators / vlm_backend | framework_version / evaluator_versions / backend（§3.4） |
| §7.1 CLI 覆盖参数风格 | --evaluator.compliance.limit_words | --set evaluators.compliance.limit_words=...（§7.1） |
| §9 项目结构 | 无 prompts/ 目录 | 新增 video_eval/prompts/（§6.1 Prompt 模板） |
| §7.2 配置结构 | vlm_judge.dimensions 单列表 | dimensions_main_image / dimensions_external / dimensions_general |
| §11 评估维度 | video_type 仅两种 | 新增 general（通用 AIGC，§8.2） |
| §5.4 串行加载决策 | 串行加载释放（单视频） | 单视频保持串行；批量模式 resident/sequential 可配（§4.1） |
| §4.3 多 alias 注册 | 一个组件可以注册多个 alias | v1 强制 alias == cls.name（§1.5，避免配置段与 dimension 名歧义） |
| §4.4 LRU cache | 物化后 LRU cache 缓住 | 无 LRU——物化结果直接替换 `_entries` 条目，`_broken` 负向缓存防重试（§1.6） |
| §4.1 Registry 字段名 | `_objs` / `_name` | `_entries` / `_meta` / `_broken` / `_origins` / `_frozen`（§1.1） |
| §7.2 配置结构 | VLM 后端参数写在 vlm_judge 段（model / api_*） | 新增独立 `backends.<name>` 段，评估器段只保留 `backend` 选择键（§2.2 / §7.3，决策 D7） |
| §5.5 降级哲学 | 依赖缺失 → 相关维度 unavailable，不报错 | 维持该哲学，但**强否决维度例外**：其依赖不可满足时在启动前置阶段 fail fast（退出码 2），不静默降级也不转 REJECT（§5.3，决策 D5） |
| §5.5 抽取器失败分类 | 无 required/optional 概念 | 抽取器自声明 `criticality`（默认 optional），框架不维护内置名单（§2.3 / §6.6，决策 D6） |
| §6 评分汇总 | 否决维度不参与加权 | 仅**纯否决**维度（有阈值无权重，v6 起含 default_weights 口径）不参与加权；阈值与权重双声明的维度既否决也计分（§5.1） |
| §7.2 配置结构 | 抽取器无开关 | 抽取器支持 `extractors.<name>.enabled` 禁用（C4/v6，§4.5/§7.3） |
| §6 VLM 评估器依赖 | VLMJudge 硬依赖 frames+asr+ocr | VLMJudge 仅硬依赖 `["frames"]`；ASR/OCR 为子维度级软依赖，缺失时降级评估而非整体跳过（v7/A2，§6.1） |
| §5.1 融合策略接口 | fuse() 三参数 | fuse() 增加 `default_weights` 第四参数，Pipeline 预计算后注入（v7/A1，§2.4） |
