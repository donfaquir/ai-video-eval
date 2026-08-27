# video-eval 设计方案

## 一、项目定位

**video-eval**：一个开源的 AI 生成视频质量评估框架。

核心价值不是"打包一堆算法"，而是提供一个插件化的评估管线，让社区可以按需组合评估维度，甚至贡献自己的评估器。类似于 ESLint 之于代码检查——框架管流程和接口，规则/插件由社区贡献。

**一句话定位**：给框架一个视频 + 商品元信息，它返回结构化的多维度评分 + 证据 + 改进建议，评估维度可插拔。

**目标场景**：电商 AI 生成视频质量评估（商品主图视频 + 外投引流视频），同时兼容通用 AIGC 视频质量评估。

## 二、架构参考

借鉴两个成熟框架的架构模式：

**lm-evaluation-harness（EleutherAI, ~7k star）**：LLM 评估领域的事实标准。其 Registry + 装饰器注册 + 懒加载机制、多 Registry 分离、DummyLM mock 模式、配置三层优先级、可复现性 meta 信息等模式经过多轮迭代验证，直接照搬其设计模式（不是代码，是模式）。

**DeepSeek Harness（deepseek-ai, ~38k star）**：Agent 框架，"一切皆插件"哲学。其依赖驱动激活（插件声明 requires，服务可用时自动激活）、注册即副作用 + 完整清理（fiber 生命周期）、能力缝隙模式（接口/提供者/消费者三层分离）等思路作为长期方向吸收。

两者的共同启示：**框架核心只管流程和接口，所有能力（评估器、特征抽取器、模型后端、融合策略）都是可注册、可替换、可发现的插件。**

## 三、核心架构

三层分离，每层职责单一：

```
输入层 → 管线层 → 输出层

输入层：视频文件 + 商品元信息（标题、卖点、主图、视频类型）
  │
  ▼
管线层（框架核心）
  ├── 特征抽取（一次性，所有评估器共用）
  │     ├── VideoMeta   抽帧、分辨率、码率、转场检测
  │     ├── ASR         口播转写 + 时间戳
  │     ├── OCR         屏幕花字 + 时间戳
  │     └── VisualFeat  CLIP/SigLIP 视觉特征编码
  │
  ├── 评估器（插件，按需懒加载）
  │     ├── 技术画质    规则 + OpenCV
  │     ├── AIGC瑕疵    小模型
  │     ├── 商品还原    SigLIP 图文相似
  │     ├── 合规审查    敏感词 + NER
  │     ├── VLM裁判     Qwen3-VL / Gemini API / Mock
  │     └── [社区插件...]  任何人可贡献
  │
  └── 融合决策
        ├── 加权总分（按视频类型不同权重）
        ├── 一票否决
        └── 建议生成
  │
  ▼
输出层：结构化 JSON（分数 + 证据 + 建议 + 等级 + meta）
```

关键设计决策：**特征抽取只做一次**。抽帧、ASR、OCR 的结果打包成一个 EvalContext 对象，所有评估器都消费这个对象，不重复解码视频。这是性能的生死线。

## 四、Registry 系统（核心）

参考 lm-eval-harness 的 Registry 模式，这是整个插件系统的基座。

### 4.1 Registry 类

`Registry[T]` 是一个泛型、线程安全的映射表：alias → 对象或字符串占位符。核心字段：

- `_name`：人类可读的名称，用于错误信息
- `_base_cls`：可选的基类，注册时做类型校验
- `_objs`：alias → 对象 或 Placeholder（字符串路径）
- `_lock`：threading.RLock，保护并发注册

### 4.2 多 Registry 分离

不同类型的组件用不同的 Registry，独立管理：

| Registry 变量 | 存什么 | 示例 |
|---|---|---|
| `evaluator_registry` | 评估器插件类 | TechnicalQualityEvaluator, VLMJudge |
| `extractor_registry` | 特征抽取器类 | ASRExtractor, OCRExtractor |
| `backend_registry` | VLM 后端类 | LocalBackend, APIBackend, MockBackend |
| `fusion_registry` | 融合策略类 | WeightedFusion, VetoFusion |

### 4.3 装饰器注册

用装饰器在类/函数定义时注册，零额外代码：

```python
@register_evaluator("technical_quality")
class TechnicalQualityEvaluator(BaseEvaluator): ...

@register_evaluator("vlm_judge")
class VLMJudge(BaseEvaluator): ...

@register_backend("api")
class APIBackend(BaseBackend): ...

@register_backend("mock")
class MockBackend(BaseBackend): ...
```

一个组件可以注册多个 alias（如一个评估器同时注册 `technical_quality` 和 `quality`）。

### 4.4 懒加载

参考 lm-eval-harness 的 Placeholder 机制。注册时可以只存一个字符串路径（`"video_eval.evaluators.technical_quality:TechnicalQualityEvaluator"`），不真正 import。第一次 `registry.get("technical_quality")` 时才触发 import，之后 LRU cache 缓住。

三层收益：
- CLI 启动快（几十个评估器不全量 import torch/whisper/open_clip）
- 可选依赖管理（没装 faster-whisper 时 ASR 相关评估器标记 unavailable，不报错）
- 防止循环 import（Registry 可先注册占位符，模块后续再加载）

### 4.5 冲突检测 + 智能提示

**占位符 → 具体对象升级允许**：如果一个 alias 已经存了字符串占位符，后来看正主注册了，允许替换。

**具体对象 → 具体对象冲突拒绝**：如果已经是具体对象，又来一个同名注册，直接报错——防止静默覆盖。

**智能错误提示**：`get("tehcincial")` 找不到时，用前缀和子串匹配建议 `"did you mean 'technical_quality'?"`。对开源项目的用户体验很重要——别人第一次用打错名字，一个好提示就留住了他。

### 4.6 Freeze 与 Origin 追踪

- `freeze_all()`：初始化完成后锁定 Registry，防止运行时意外注册
- `origin()`：返回组件定义的源文件位置，调试时定位"这个评估器从哪来的"

## 五、插件系统设计

### 5.1 评估器插件接口

每个评估器继承 `BaseEvaluator`，实现以下接口：

```python
class BaseEvaluator(ABC):
    # 元信息（类属性，声明式）
    name: str                    # 插件名，如 "technical_quality"
    version: str = "0.1.0"
    device_requirement: str = "any"  # "cuda" / "mps" / "any"
    requires: list[str] = []     # 依赖的 context 字段，如 ["asr", "ocr"]
    config_schema: dict = {}     # 配置项的 JSON Schema，用于验证用户 config

    # 生命周期方法
    def __init__(self, device_manager: DeviceManager, config: dict):
        """初始化：加载模型、读取配置。框架调用。"""
        ...

    def evaluate(self, context: EvalContext) -> EvalResult:
        """评估：接收上下文，返回结果（分数 + 证据 + 建议）。"""
        ...

    def __enter__(self):
        """Context manager 入口：等同于 init。保证 cleanup 一定被调用。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 出口：释放模型、释放显存。即使中途异常也执行。"""
        ...
```

**requires 声明**（借鉴 DeepSeek Harness 的依赖驱动激活）：评估器声明它需要哪些 context 字段。如果用户没装 whisper（ASR 不可用），依赖 ASR 的评估器自动跳过，不崩溃。比"检查 + 警告"更优雅——是自动跳过而非手动配置。

**Context manager 协议**（借鉴 DeepSeek Harness 的"注册即副作用 + 完整清理"）：框架用 `with` 语句调用评估器，保证 `__exit__` 一定执行，即使中途异常也释放显存。比裸的 init/cleanup 方法对更安全——Python 的 context manager 语义保证了这一点。

### 5.2 VLM 后端：能力缝隙模式

借鉴 DeepSeek Harness 的能力缝隙（Capability Seam）模式，把 VLM 后端拆成三层：

**接口（Service Definition）**：`BaseBackend` 抽象基类，定义 `judge(video, prompt, context) -> VLMResult` 接口。

**提供者（Provider）**：具体实现，通过 `@register_backend` 注册：
- `LocalBackend`：加载 Qwen3-VL 到本地设备（CUDA/MPS）
- `APIBackend`：调用 Gemini3-Flash / GPT-5.2 API
- `MockBackend`：返回固定分数，CI 和调试用

**消费者（Consumer）**：`VLMJudge` 评估器。它不关心后端是哪个，只调 `backend.judge()`。加一个新 API 后端（如 Claude）只需要写新的 provider 并注册，不动消费者代码。

```python
class BaseBackend(ABC):
    @abstractmethod
    def judge(self, video_path: str, prompt: str, context: dict) -> VLMResult:
        ...

@register_backend("local")
class LocalBackend(BaseBackend): ...

@register_backend("api")
class APIBackend(BaseBackend): ...

@register_backend("mock")
class MockBackend(BaseBackend): ...
```

后端选择通过 config 或环境变量切换，零代码改动：
```yaml
vlm_judge:
  backend: api    # local / api / mock
```
或 `VIDEO_EVAL_VLM_BACKEND=api`。

### 5.3 插件发现机制

两层并行：

**内置插件**：放在 `video_eval/evaluators/` 目录下。框架启动时扫描该目录，自动注册所有带 `@register_evaluator` 装饰器的类。类似 lm-eval-harness 扫描 `lm_eval/tasks/` 目录。

**外部插件**：通过 Python entry_points 机制。社区开发 `video-eval-plugin-watermark` 独立包，在 `pyproject.toml` 声明：

```toml
[project.entry-points."video_eval.evaluators"]
watermark = "video_eval_plugin_watermark:WatermarkEvaluator"
```

框架启动时用 `importlib.metadata.entry_points()` 扫描 `video_eval.evaluators` group，自动发现并注册外部插件。类似 pytest 的插件机制。

### 5.4 插件生命周期

```
框架启动
  │
  ├── 1. 初始化 Registry（注册占位符，不 import 重依赖）
  │
  ├── 2. 扫描插件
  │     ├── 扫描 video_eval/evaluators/ 目录（内置）
  │     ├── 扫描 entry_points（外部）
  │     └── 全部注册到 evaluator_registry（仍是占位符）
  │
  ├── 3. 过滤与检查
  │     ├── 按 config 中 enabled 过滤
  │     ├── 检查 device_requirement（cuda 插件在 cpu 上 → 跳过 + 警告）
  │     └── 检查 requires（需要 asr 但 asr 不可用 → 跳过 + 警告）
  │
  ├── 4. 特征抽取（一次性，产出 EvalContext）
  │
  ├── 5. 逐个评估
  │     for evaluator_cls in enabled_evaluators:
  │         with evaluator_cls(device_manager, config) as evaluator:  # 懒加载 + context manager
  │             result = evaluator.evaluate(context)
  │             results[dim] = result
  │         # __exit__ 自动释放显存
  │
  ├── 6. 融合决策 → 输出
  │
  └── 7. Registry freeze（防止运行时意外注册）
```

**为什么串行不并行**：单卡显存有限。Qwen3-VL 8B 16GB + SigLIP 2GB + Whisper 3GB 已接近 40GB 上限。串行加载、用完释放（context manager 保证）是最稳妥的。多卡场景以后再优化。

### 5.5 降级策略

如果某个插件因设备不可用或依赖缺失被跳过：
1. 打印警告（含跳过原因）
2. 在结果里标记该维度为 `"status": "skipped"` 而非 `"scored"`
3. 融合决策时自动调整权重（把跳过的维度权重按比例分摊到其他维度）
4. 输出 JSON 的 `meta.skipped` 字段记录所有被跳过的维度及原因

## 六、设备适配策略

### 6.1 DeviceManager

框架内置设备管理器，启动时自动检测：

- 有 CUDA → `torch.device("cuda")` + `float16`
- 无 CUDA 但有 MPS（Apple Silicon）→ `torch.device("mps")` + `float16`
- 都没有 → `torch.device("cpu")` + `float32`

所有插件通过 DeviceManager 获取设备，不自己调 `torch.device()`。这样同一份代码在 Mac 和 GPU 服务器上都能跑。

DeviceManager 还提供 `can_load_model(param_count)` 方法，粗略判断是否有足够显存/内存加载模型。评估器在 `__init__` 里可以检查，如果显存不够就主动 raise SkipError，框架捕获后走降级策略。

### 6.2 三种 VLM 后端

**本地模型模式（local）**：加载 Qwen3-VL 到本地设备（CUDA 或 MPS）。Mac 上能调试，但慢（M2 Max 32GB 可以跑，速度约 CUDA 的 1/3）。

**API 代理模式（api）**：调用 Gemini3-Flash / GPT-5.2 API。Mac 上调试时用这个模式，快、不吃本地显存。通过环境变量切换：`VIDEO_EVAL_VLM_BACKEND=api` + `GEMINI_API_KEY=xxx`。

**Mock 模式（mock）**：返回固定分数。CI 测试和纯流程调试用，不需要任何模型。借鉴 lm-eval-harness 的 DummyLM 模式。

三种模式实现同一个 `BaseBackend` 接口，切换时零代码改动——只改环境变量或 config。

## 七、配置系统

### 7.1 三层优先级

参考 lm-eval-harness 的配置分层：

1. **CLI 参数**（最高优先级）：`--evaluator.compliance.limit_words "最,第一"` 临时覆盖
2. **配置文件**：`config.yaml`，提供基础参数
3. **插件默认值**（最低优先级）：评估器类属性 `config_schema` 声明的默认值

高优先级覆盖低优先级。用户调试时可以临时覆盖某个参数而不改配置文件。

### 7.2 配置结构

```yaml
device:
  preferred: auto          # auto / cuda / mps / cpu

extractors:
  fps: 1
  max_frames: 64
  asr:
    model_size: large-v3
  ocr:
    confidence: 0.5

evaluators:                 # 每个插件一个配置段
  technical_quality:
    enabled: true
  compliance:
    enabled: true
    limit_words: [最, 第一, 国家级]
  vlm_judge:
    enabled: true
    backend: api           # local / api / mock
    model: gemini-3-flash  # API 模式时填模型名；local 模式填 Qwen3-VL-8B-Instruct
    dimensions:            # 启用哪些 VLM 维度
      - sellpoint_coverage
      - cross_modal
      - hook_strength

fusion:
  strategy: weighted_veto  # 融合策略，注册到 fusion_registry
  weights_main_image:
    technical_quality: 0.15
    product_fidelity: 0.20
    compliance: veto       # 一票否决
  weights_external:
    hook_strength: 0.20
```

每个插件的配置段是独立的，框架把对应 YAML 段传给插件的 `__init__`。插件用 `config_schema` 声明期望的配置项和类型，框架在启动时做一次校验——配置不对就 fail fast，不等到跑到一半才报错。

### 7.3 配置校验

参考 lm-eval-harness 的 `lm-eval validate` 命令。`video-eval config check` 扫描所有已启用插件的 config_schema，校验：
- 必填项是否缺失
- 类型是否正确
- 依赖的提取器是否可用（vlm_judge 需要 ASR，asr 提取器是否安装）

## 八、输出格式与可复现性

### 8.1 结构化输出

```json
{
  "video_path": "./sample.mp4",
  "video_type": "external",
  "overall_score": 0.72,
  "grade": "B",
  "passed": true,
  "veto_reasons": [],
  "dimension_results": {
    "technical_quality": {"score": 0.85, "status": "scored", "evidence": "..."},
    "compliance": {"score": 1.0, "status": "scored", "evidence": "未检测到合规风险"},
    "hook_strength": {"score": 0.75, "status": "scored", "evidence": [...]}
  },
  "suggestions": [
    "[marketing_logic] 视频缺少明确的痛点引入环节，建议在前3秒加入用户痛点场景"
  ],
  "meta": {
    "video_eval_version": "0.1.0",
    "device": "mps",
    "evaluators": {
      "technical_quality": "0.1.0",
      "compliance": "0.1.0",
      "vlm_judge": "0.1.0"
    },
    "vlm_backend": "api",
    "vlm_model": "gemini-3-flash",
    "skipped": [],
    "config_hash": "a1b2c3d4",
    "timestamp": "2026-08-26T15:00:00+08:00"
  }
}
```

### 8.2 可复现性

参考 lm-eval-harness 的可复现性设计。`meta` 字段记录：
- video-eval 版本 + 各插件版本
- 设备信息（cuda/mps/cpu）
- VLM 后端和模型名
- 被跳过的维度及原因
- config 文件的 hash（配置变更可追溯）
- 评估时间戳

这样别人复现你的评估结果时有据可查，也方便排查"为什么这次跑的分和上次不一样"——大概率是 config 变了或后端换了。

## 九、项目结构

```
video-eval/
├── README.md
├── LICENSE                        # Apache-2.0
├── pyproject.toml                 # 项目元信息 + 依赖 + entry_points
├── config.yaml.example            # 配置模板
│
├── video_eval/                    # 主包
│   ├── __init__.py
│   ├── cli.py                      # CLI 入口（click 框架）
│   ├── config.py                  # 配置加载（三层优先级合并）
│   │
│   ├── core/                      # 框架核心（不含任何算法）
│   │   ├── __init__.py
│   │   ├── registry.py             # Registry[T] 泛型类 + 装饰器
│   │   ├── base.py                 # BaseEvaluator / BaseBackend / BaseExtractor 基类
│   │   ├── pipeline.py             # 管线编排（抽取→评估→融合）
│   │   ├── context.py              # EvalContext 数据结构
│   │   ├── device.py               # DeviceManager（CUDA/MPS/CPU）
│   │   ├── schemas.py              # Pydantic 输入输出模型
│   │   └── fusion.py               # 融合决策（加权+否决+建议）
│   │
│   ├── backends/                   # VLM 后端（能力缝隙的 provider 层）
│   │   ├── __init__.py
│   │   ├── local.py                # 本地模型后端（Qwen3-VL）
│   │   ├── api.py                  # API 后端（Gemini/GPT）
│   │   └── mock.py                 # Mock 后端（固定分数）
│   │
│   ├── extractors/                 # 特征抽取器（内置，注册到 extractor_registry）
│   │   ├── __init__.py
│   │   ├── video_meta.py
│   │   ├── asr.py
│   │   ├── ocr.py
│   │   └── clip_features.py
│   │
│   └── evaluators/                # 内置评估器（注册到 evaluator_registry）
│       ├── __init__.py             # 扫描并注册本目录所有评估器
│       ├── technical_quality.py
│       ├── aigc_defect.py
│       ├── product_fidelity.py
│       ├── compliance.py
│       └── vlm_judge.py
│
├── docker/
│   ├── Dockerfile.cpu              # CPU/MPS 镜像（Mac 调试用）
│   ├── Dockerfile.gpu              # CUDA 镜像（生产部署用）
│   ├── docker-compose.yml
│   └── entrypoint.sh
│
├── docs/
│   ├── design.md                   # 本文档
│   ├── competitor-analysis.md      # 竞品分析
│   ├── plugin-development.md       # 插件开发指南
│   ├── docker-deploy.md
│   └── dimensions.md               # 维度说明
│
├── examples/
│   ├── single_eval.py              # 单视频评估示例
│   └── batch_eval.py               # 批量评估示例
│
├── tests/
│   ├── test_registry.py            # Registry 注册/懒加载/冲突检测
│   ├── test_pipeline.py            # 管线端到端（用 MockBackend）
│   └── test_device.py              # 设备检测
│
└── plugins/                        # 社区插件示例（非主包）
    └── watermark_detect/
        └── README.md
```

**分层原则**：`core/` 只管流程和接口，不含任何算法实现。`backends/` 是 VLM 后端的 provider 层。`extractors/` 和 `evaluators/` 是内置实现，可以独立替换。社区插件不放在主包里，通过 entry_points 注册。

**与之前版本的变更**：新增 `core/registry.py`（独立的 Registry 模块）、新增 `backends/` 目录（VLM 后端从 vlm_judge.py 拆出，独立成层）、`core/base.py` 独立存放所有基类。

## 十、CLI 设计

第一版只做 CLI，后续加 Web UI。

```
# 评估单个视频
video-eval eval \
  --video ./sample.mp4 \
  --product-title "便携蓝牙音箱" \
  --selling-points "防水IPX7" "12小时续航" \
  --product-images ./product_1.jpg ./product_2.jpg \
  --video-type external \
  --output ./result.json

# 批量评估（目录扫描 + manifest）
video-eval batch \
  --input ./videos/ \
  --manifest ./manifest.csv \
  --output ./results/

# 列出所有已注册的评估器（含可用性状态）
video-eval plugins

# 查看某个评估器的详情和配置 schema
video-eval plugins --detail vlm_judge

# 验证配置文件（检查必填项、类型、依赖可用性）
video-eval config check

# 查看设备状态
video-eval device info
```

CLI 用 click 框架。后续加 Web UI 时，CLI 的逻辑层直接复用，只加一层 HTTP API + 前端。

## 十一、评估维度设计

### A. 共享基础维度（主图 & 外投都要）

| 维度 | 信号来源 | 打分方式 |
|------|---------|---------|
| 技术画质 | 分辨率/码率/模糊/闪烁/黑边 | 规则 + OpenCV |
| 时序连贯性 | 相邻帧一致性 | DINOv3/CLIP 帧间相似度 |
| AIGC 瑕疵 | 手指/文字扭曲、脸崩 | 专用小模型 |
| 商品还原度 | 视频帧 vs 商家主图 | SigLIP 图文相似 |
| 合规审查 | 极限词/医疗词/敏感实体 | OCR + ASR + 敏感词库 |
| 跨模态一致性 | 口播/花字 vs 画面 | VLM 裁判（E-VAds CM 任务） |

### B. 主图视频专属维度

| 维度 | 说明 |
|------|------|
| 卖点覆盖率 | 从商品标题/卖点列表抽取，检查每个卖点是否被覆盖 |
| 商品出镜时长 | 商品在画面中出现的秒数占比 |
| 主图一致性 | 视频首帧与商家主图的一致性 |

### C. 外投引流视频专属维度

| 维度 | 说明 |
|------|------|
| 黄金前 3 秒钩子 | 前 3 秒是否有强视觉冲击/悬念/利益点 |
| CTA 明确性 | 是否有"点击下方链接"等明确 CTA |
| 目标受众匹配 | 视频风格与商品目标受众的匹配度 |
| 营销逻辑完整性 | 痛点→解决方案→证据→引导的完整链路 |

## 十二、Docker 部署策略

两套 Dockerfile，共享同一个代码镜像层，只差基础镜像和依赖：

**Dockerfile.cpu（Mac / CI 调试用）**：基于 `python:3.11-slim`。装 CPU 版 PyTorch。VLM 裁判默认走 API 模式。镜像体积小（约 1.5GB），Mac 上 Docker Desktop 无压力。

**Dockerfile.gpu（生产部署用）**：基于 `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`。装 CUDA 版 PyTorch。VLM 裁判走本地模型模式（Qwen3-VL 8B）。镜像体积大（约 6-8GB），需要 `--gpus all` 运行。

**docker-compose.yml**：

```yaml
services:
  video-eval:
    build:
      context: .
      dockerfile: docker/Dockerfile.cpu    # Mac 调试
      # dockerfile: docker/Dockerfile.gpu   # 生产部署时切换
    volumes:
      - ./data:/app/data
      - ./config.yaml:/app/config.yaml
      - ./results:/app/results
    environment:
      - VIDEO_EVAL_VLM_BACKEND=api   # Mac 用 API；GPU 机器改 local
      - GEMINI_API_KEY=${GEMINI_API_KEY}
```

Mac 调试时 `Dockerfile.cpu` + VLM 走 API；部署到 GPU 机器时切到 `Dockerfile.gpu` + VLM 走 local，改两行配置，代码不动。

## 十三、开发阶段规划

### Phase 1：骨架跑通（1-2 周）

目标：Mac 上能跑通端到端流程，用 Mock 后端验证框架。

- 搭项目结构 + pyproject.toml + CLI 骨架
- 实现 Registry 类（注册/懒加载/冲突检测/智能提示）
- 实现 BaseEvaluator / BaseBackend 基类 + 装饰器
- 实现 DeviceManager（CUDA/MPS/CPU 检测）
- 实现 Pipeline 编排（抽取 → 评估 → 融合）
- 实现 MockBackend + 2 个最简评估器（技术画质 + 合规审查，纯规则无模型）
- 实现配置三层优先级合并
- 特征抽取层（ffprobe + 抽帧，ASR/OCR 暂留接口）
- 融合决策 + JSON 输出（含 meta 字段）
- `video-eval plugins` / `video-eval config check` / `video-eval device info` 命令
- Dockerfile.cpu 能构建 + 运行
- 基础测试：test_registry + test_pipeline（用 MockBackend）+ test_device

### Phase 2：核心评估器（2-3 周）

目标：传统评估器全部就绪，Mac MPS 上能跑。

- 接入 Whisper ASR（注册到 extractor_registry，MPS 支持）
- 接入 OCR
- 接入 SigLIP 商品还原度（注册到 evaluator_registry）
- 接入 AIGC 瑕疵检测（v0 先用 CLIP zero-shot）
- 5% 人工校准流程设计

### Phase 3：VLM 裁判（2-3 周）

目标：VLM 三种后端都跑通。

- 实现 LocalBackend（Qwen3-VL，注册到 backend_registry）
- 实现 APIBackend（Gemini3-Flash / GPT-5.2）
- 完善 MockBackend
- 5 个维度的 Prompt 模板
- 输出解析器（JSON 提取 + 容错）
- Dockerfile.gpu 构建测试（需要在 GPU 机器上验证）

### Phase 4：开源准备（1-2 周）

- README + 文档 + 插件开发指南
- LICENSE（Apache-2.0）
- CONTRIBUTING.md + issue/PR 模板
- examples 目录 + 示例
- GitHub Actions CI（lint + test + Docker build）
- 发 v0.1.0 release

### Phase 5：Web UI（后续）

- FastAPI 包装 CLI 逻辑层
- 前端：视频上传 + 结果展示 + 证据可视化
- Docker compose 加前端服务

## 十四、关键设计决策的理由

**为什么用 Registry + 装饰器而不是目录扫描**：目录扫描只能发现文件位置，不能做类型校验、冲突检测、懒加载、智能提示。Registry 模式把"发现 + 注册 + 校验 + 懒加载"统一管理，是 lm-eval-harness 验证过的方案。

**为什么懒加载**：video-eval 有几十个评估器，每个可能依赖 torch、whisper、open_clip、vbench 等重包。全量 import 会让 CLI 启动慢 10+ 秒，而且用户没装某个包就直接报错。懒加载注册时只存字符串，用到才 import——CLI 启动 <1 秒，没装 whisper 只影响 ASR 相关评估器，不影响框架本身。

**为什么 VLM 后端独立成层（能力缝隙模式）**：如果后端选择写死在 VLMJudge 里，加一个 Claude API 后端就要改 VLMJudge 的代码。拆成 BaseBackend 接口 + provider 实现后，加新后端只需写一个新类并 `@register_backend("claude")`，不动 VLMJudge 一行代码。这借鉴了 DeepSeek Harness 的能力缝隙模式。

**为什么用 context manager 而非裸 init/cleanup**：Python 的 `with` 语义保证 `__exit__` 一定执行，即使 `evaluate()` 中途抛异常。裸的 cleanup() 没有这个保证——如果评估器崩了，cleanup 不会被调用，显存泄漏。借鉴 DeepSeek Harness 的"注册即副作用 + 完整清理"理念，但用 Python 原生的 context manager 实现，比 dsh 的 fiber 机制简单得多。

**为什么插件串行加载而不是并行**：单卡显存有限。Qwen3-VL 8B 16GB + SigLIP 2GB + Whisper 3GB 已接近 40GB 上限。串行加载、用完释放（context manager 保证）是最稳妥的。多卡场景以后再优化。

**为什么特征抽取不做插件化**：特征抽取是所有评估器共用的基础设施，且只有固定的几种（帧、ASR、OCR、CLIP 特征）。做插件化反而增加复杂度，收益不大。但抽取器仍然注册到 extractor_registry，方便替换实现（如把 Whisper 换成 Paraformer）。

**为什么不用 LangChain / LlamaIndex 编排**：这些框架适合做 RAG 和 Agent，不适合做结构化的多模型评估管线。自己写 Pipeline 控制力更强，依赖更少，也更适合开源（别人不需要先学一个框架）。

**为什么 VLM 裁判支持 API 后端**：Mac 上跑 8B 模型虽然能跑但慢，调试体验差。API 模式让 Mac 调试时秒级返回，开发节奏快很多。而且有些用户可能不想自己部署大模型，直接用 API。

**为什么用 Apache-2.0 而不是 MIT**：Apache-2.0 有专利保护条款，对企业用户更友好，更容易吸引公司贡献。MIT 也可以，但 Apache 在 AI/ML 社区更主流（PyTorch、HuggingFace 都用 Apache）。

**为什么可以集成 VBench 而不是重复实现**：VBench 的技术画质模块（时序闪烁、主体一致性等）可以直接作为 video-eval 的一个内置评估器接入，不用自己重新实现。框架管流程，别人的指标做插件——这正好体现插件架构的价值。

**为什么输出要带 meta 字段**：参考 lm-eval-harness 的可复现性设计。评估系统最怕"分数不可复现"——同样的视频跑两次分不一样，用户就不信任。meta 字段记录版本、设备、后端、配置 hash，让每次评估都可追溯。
