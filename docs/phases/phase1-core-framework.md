# Phase 1：核心框架骨架

## 目标

Mac 上端到端跑通评估流程，用 MockBackend 验证框架骨架。产出一个可运行的 CLI 工具，能对单个视频执行最简评估并输出结构化 JSON。

## 前置依赖

无（首个阶段）。

## 交付范围

### 1. 项目结构与构建

```
video-eval/
├── pyproject.toml              # 包元数据、依赖、entry-points
├── video_eval/
│   ├── __init__.py
│   ├── cli.py                  # click CLI 入口
│   ├── core/
│   │   ├── __init__.py
│   │   ├── registry.py         # Registry[T] + Placeholder + 异常
│   │   ├── base.py             # BaseEvaluator / BaseBackend / BaseExtractor / BaseFusion
│   │   ├── schemas.py          # EvalContext / EvalResult / FinalReport / ...
│   │   ├── device.py           # DeviceManager
│   │   ├── pipeline.py         # Pipeline 编排
│   │   ├── config.py           # ConfigLoader
│   │   └── exceptions.py       # 框架级异常（ExtractionError 等）
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── technical_quality.py
│   │   └── compliance.py
│   ├── extractors/
│   │   ├── __init__.py
│   │   └── video_meta.py       # 抽帧 + 元信息（ffprobe/ffmpeg）
│   ├── backends/
│   │   ├── __init__.py
│   │   └── mock.py
│   ├── fusions/
│   │   ├── __init__.py
│   │   └── weighted_veto.py
│   └── prompts/                # 空目录，Phase 3 填充
├── tests/
│   ├── test_registry.py
│   ├── test_pipeline.py
│   ├── test_device.py
│   ├── test_fusion.py
│   ├── test_cli.py
│   └── test_config.py
└── config.yaml.example
```

### 2. Registry 系统（详设 §1 全节）

| 组件 | 说明 |
|------|------|
| `Registry[T]` 泛型类 | 含 `_entries`/`_meta`/`_broken`/`_origins`/`_frozen`/`_lock`；完整方法表（register/get/get_meta/has/list/freeze/...） |
| `Placeholder[T]` | 懒加载占位符 + 物化逻辑（8 步） |
| `PluginMeta` | 注册时同步提取的元数据快照（详设 §1.2 全字段） |
| 4 个 Registry 实例 | evaluator/extractor/backend/fusion |
| 装饰器 API | `@register_evaluator` / `@register_backend` / `@register_extractor` / `@register_fusion` |
| 异常家族 | RegistryError / NameNotFoundError / RegistryFrozenError / DuplicateRegistrationError / MaterializationError |
| 智能错误提示 | `_suggest_similar()` 前缀+子串+Levenshtein |
| 冲突检测 | 详设 §1.7 全组合 |
| 插件发现 | 目录扫描（builtin）+ entry_points（外部 Placeholder） |
| 初始化顺序 | 创建实例 → scan_directory → discover_entry_points → 不自动 freeze |

### 3. 基类设计（详设 §2 全节）

| 基类 | 关键契约 |
|------|---------|
| `BaseEvaluator` | `__init__`（轻量）/ `__enter__`（加载）/ `__exit__`（释放）/ `evaluate(context)` / `check_availability()` / `slots_for(video_type)` |
| `BaseBackend` | `__init__` / `__enter__` / `__exit__` / `judge(context, prompt)` |
| `BaseExtractor` | `__init__` / `__enter__` / `__exit__` / `extract(context)` + provides/requires/criticality |
| `BaseFusion` | `__init__(config)` / `fuse(results, video_type, weights, default_weights)` |
| `DeviceManager` | device/dtype/device_type + is_gpu/memory_info/can_load_model/satisfies |

### 4. 数据模型（详设 §3 全节）

所有 pydantic BaseModel：EvalContext（含 readonly 视图）/ EvalResult / VLMResult / FinalReport / ReportMeta / FusionOutcome / BatchItem / BatchItemResult / ValidationIssue。

### 5. Pipeline 编排（详设 §4）

| 方法 | 本阶段实现范围 |
|------|--------------|
| `run()` | 完整 13 步流程（物化 → 前置校验 → 过滤 → 抽取 → 评估 → 占位 → 融合 → 组装） |
| `run_batch()` | resident 模式（sequential 可留 Phase 2 实现） |
| 三重过滤 | F1/F2/F3 全实现 |
| 抽取器编排 | 依赖闭包扩展 + 拓扑排序 + 字段级降级 + 二次过滤 |
| 占位结果生成 | 单槽位 + 多槽位展开（effective_slots） |

### 6. 融合决策（详设 §5）

WeightedVetoFusion 完整实现：否决扫描（strict_veto + veto_thresholds）→ 权重计算（veto_only 排除 + default_weights 兜底）→ 等级判定 → 建议生成。

### 7. 内置插件（最小集）

| 插件 | 说明 |
|------|------|
| `video_meta` 抽取器 | provides=["video_meta","frames"]，criticality="required"。用 ffprobe 取元信息 + ffmpeg/imageio 抽帧 |
| `TechnicalQuality` 评估器 | requires=["frames"]，纯规则（分辨率/模糊度检测），验证评估器全链路 |
| `Compliance` 评估器 | requires=["asr","ocr"]，纯规则（敏感词匹配），验证 extraction_failed 降级链路 |
| `MockBackend` | 固定分数返回，验证 backend 契约 |
| `WeightedVetoFusion` | 默认融合策略 |

### 8. 配置系统（详设 §7）

- ConfigLoader：load() + merge_cli_overrides() + validate()
- 三层优先级：CLI --set > config.yaml > 插件默认值
- 校验规则全实现（详设 §7.2 列表）
- config.yaml.example 生成

### 9. CLI（详设 §8）

| 命令 | 本阶段 |
|------|--------|
| `video-eval eval` | 完整实现 |
| `video-eval batch` | 基础实现（resident 模式） |
| `video-eval plugins` | 完整实现 |
| `video-eval config check` | 完整实现（含 D9 probe） |
| `video-eval device info` | 完整实现 |

退出码全表（0-6）实现。

### 10. 测试

| 测试文件 | 覆盖点 |
|---------|--------|
| `test_registry.py` | 注册/装饰器/冲突表/freeze/_broken/智能提示/entry-point 物化 |
| `test_pipeline.py` | MockBackend 端到端/F2/F3 过滤/占位展开/strict_veto 两路 |
| `test_device.py` | satisfies 匹配矩阵 |
| `test_fusion.py` | 否决优先级/权重归一化/veto_only/default_weights 兜底 |
| `test_cli.py` | 退出码映射/--set 类型转换 |
| `test_config.py` | 三层合并/校验规则 |

## 验收标准

- [ ] `video-eval eval --video test.mp4 --video-type general` 输出合法 JSON（含 dimension_results + overall_score + grade + meta）
- [ ] `video-eval plugins` 冷启动 < 1s
- [ ] `video-eval config check` 通过（默认配置）
- [ ] compliance 因 asr/ocr 不可用时，输出 `skipped: extraction_failed`（而非崩溃）
- [ ] strict_veto_dims 中 compliance 依赖 asr 不可用时，退出码 2（前置校验拦截）
- [ ] MockBackend 端到端测试通过
- [ ] 全部单元测试通过（pytest）

## 参考详设章节

§1（Registry 全节）、§2（基类全节）、§3（数据模型全节）、§4（Pipeline 全节）、§5（融合全节）、§6.2-6.3（Compliance/TechnicalQuality）、§7（配置全节）、§8（CLI 全节）、§9.4（检查清单）、§10.1-10.2（测试+验收）
