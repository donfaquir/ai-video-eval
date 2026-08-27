# Phase 3：VLM 裁判系统

## 目标

实现 VLMJudge 评估器 + 三种后端（Local/API/Mock），完成 Prompt 模板设计，使 VLM 维度端到端可评估。

## 前置依赖

Phase 2 完成（ASR/OCR/CLIP 抽取器就绪，EvalContext 完整填充）。

## 交付范围

### 1. VLMJudge 评估器（详设 §6.1）

```python
@register_evaluator("vlm_judge")
class VLMJudge(BaseEvaluator):
    name = "vlm_judge"
    device_requirement = "any"
    backend_config_key = "backend"
    requires = ["frames"]                    # v7：仅硬依赖 frames
    dimension_slots = {
        "main_image": ["sellpoint_coverage", "cross_modal"],
        "external": ["hook_strength", "marketing_logic", "audience_match", "cross_modal"],
        "general": ["cross_modal"],
    }
    default_weights = {
        "sellpoint_coverage": 0.20, "cross_modal": 0.15,
        "hook_strength": 0.20, "marketing_logic": 0.15, "audience_match": 0.10,
    }
```

实现要点：
- **子维度级软依赖**（v7/A2）：ASR/OCR 为软依赖，evaluate() 内按实际可用性组装 prompt
- **cross_modal**：需 ASR 或 OCR 至少一种，两者均缺 → skip: missing_dependency
- **hook_strength/marketing_logic/audience_match**：纯视觉评估，ASR/OCR 可用时辅助注入
- **sellpoint_coverage**：需 product_info，缺失 → skip: missing_product_info
- **Backend 生命周期**：VLMJudge.__enter__ 内获取并启动 backend
- **调用编排**：api 后端按 api_concurrency 并发调用子维度；local 串行
- **失败预算**：api_max_failures 超出 → 剩余子维度直接占位
- **evidence.input_modalities**：标注实际输入的模态集合

### 2. LocalBackend（Qwen3-VL）

```python
@register_backend("local")
class LocalBackend(BaseBackend):
    name = "local"
    device_requirement = "gpu"
```

实现要点：
- `__enter__`：加载 Qwen3-VL 模型（transformers + torch）
- `judge(context, prompt)`：消费 context.frames 做推理
- 五级评分映射：`score = (level - 1) / 4`
- 输出解析：正则提取 JSON + 字段校验
- D8 缓存：已归一化帧 tensor 按 video_path 缓存（容量=1）
- 配置：backends.local.model

### 3. APIBackend（Gemini/OpenAI）

```python
@register_backend("api")
class APIBackend(BaseBackend):
    name = "api"
    device_requirement = "any"
```

实现要点：
- `__enter__`：验证 API key 环境变量
- `judge(context, prompt)`：上传 video_path 或 frames → 调用 API
- D8 缓存：已上传 file reference 按 video_path 缓存（容量=1，线程安全）
- 重试：指数退避（retry_base / max_retries / timeout）
- 输出解析：同 LocalBackend
- `__exit__`：清空缓存（API 侧文件如需删除，在此完成）
- 配置：backends.api（provider/model/timeout/max_retries/retry_base）
- API key 只从环境变量读取

### 4. MockBackend 增强

Phase 1 已有基础版本，本阶段增强：
- 支持按维度返回不同固定分数（便于测试否决/等级边界）
- 支持延迟模拟（测试超时/并发）
- 输出格式与真实 backend 一致

### 5. Prompt 模板（5 个子维度）

```
video_eval/prompts/
├── sellpoint_coverage.txt
├── cross_modal.txt
├── hook_strength.txt
├── marketing_logic.txt
└── audience_match.txt
```

每个模板：
- E-VAds 五级评分标准
- 证据溯源要求（时间戳 + 模态标注）
- 结构化输出格式说明（JSON schema）
- 自适应段落（ASR/OCR 可用时的模态说明，不可用时的替代指令）

### 6. VLMOutputParseError

框架内置异常，携带 raw_output 片段。backend 解析失败时抛出，VLMJudge 捕获并标记该子维度 `error: parse_failed`。

### 7. 测试

| 测试 | 覆盖点 |
|------|--------|
| `test_vlm_judge.py` | 子维度软依赖（ASR 缺/OCR 缺/两者缺）/product_info 缺失/evidence.input_modalities/prompt 自适应 |
| `test_backends.py` | 五级映射/解析容错/重试/失败预算/D8 上传缓存/线程安全 |
| `test_local_backend.py` | Qwen3-VL 加载/推理（需 GPU，CI 可 skip）|
| `test_api_backend.py` | API 调用 mock/重试逻辑/超时处理 |

## 验收标准

- [ ] `video-eval eval --video test.mp4 --video-type external --set evaluators.vlm_judge.backend=mock`：输出 4 个 VLM 子维度得分
- [ ] `video-eval eval --video test.mp4 --video-type external --set evaluators.vlm_judge.backend=api`（需 API key）：端到端返回真实 VLM 评分
- [ ] ASR 不可用时：hook_strength/marketing_logic/audience_match 仍正常评分（prompt 无 ASR 段）
- [ ] ASR+OCR 均不可用时：cross_modal skip（missing_dependency），其余子维度正常
- [ ] api_max_failures 超出：剩余子维度直接 error: evaluation_failed 占位
- [ ] D8：同一视频 4 个子维度只上传/预处理 1 次（日志可验证）
- [ ] 单视频端到端延迟：mock < 30s；api < 120s（external 4 次 VLM 调用）

## 参考详设章节

§2.2（BaseBackend）、§6.1（VLMJudge 全节）、§3.3（VLMResult）、§2.2 judge() 契约、§2.2 五级映射、§2.2 APIBackend 配置
