# Phase 3 Spec 01：Prompt 模板

## 目标

设计 5 个子维度的 VLM Prompt 模板，供 VLMJudge 调用 backend 时使用。模板遵循 E-VAds 五级评分格式，支持自适应模态段落。

## 产出文件

```
video_eval/prompts/
├── sellpoint_coverage.txt
├── cross_modal.txt
├── hook_strength.txt
├── marketing_logic.txt
└── audience_match.txt
```

## 模板规范

每个模板包含以下结构：

```
[ROLE]
You are a professional video quality evaluator...

[TASK]
Evaluate the {dimension_name} of this video...

[INPUT_MODALITIES]
{dynamic: based on available modalities}

[SCORING_CRITERIA]
5-level rubric (1-5):
  Level 5: ...
  Level 4: ...
  Level 3: ...
  Level 2: ...
  Level 1: ...

[OUTPUT_FORMAT]
Return a JSON object:
{
  "level": <1-5>,
  "reasoning": "<explanation>",
  "evidence": [{"modality": "visual|audio|text", "timestamp": <float|null>, "detail": "<desc>"}],
  "suggestion": "<improvement advice>"
}
```

### 模板中的占位符

| 占位符 | 含义 | 由谁填充 |
|--------|------|---------|
| `{video_type}` | main_image / external / general | VLMJudge._build_prompt |
| `{product_title}` | 商品标题 | VLMJudge (仅 sellpoint_coverage) |
| `{selling_points}` | 卖点列表 | VLMJudge (仅 sellpoint_coverage) |
| `{asr_text}` | ASR 转写全文 | VLMJudge (可选，不可用时整段移除) |
| `{ocr_text}` | OCR 检测文字汇总 | VLMJudge (可选，不可用时整段移除) |
| `{modality_note}` | 可用模态说明 | VLMJudge (动态生成) |

### 自适应段落

模板中 ASR/OCR 相关段落使用条件标记：

```
{{#if asr_available}}
## Audio Transcription (ASR)
The following is the speech transcription of the video:
{asr_text}
{{/if}}

{{#if ocr_available}}
## On-screen Text (OCR)
The following text was detected on screen:
{ocr_text}
{{/if}}
```

VLMJudge 的 `_build_prompt()` 方法负责根据可用性移除不可用段落（简单字符串替换，不需要模板引擎）。

## 各维度评分标准概要

### sellpoint_coverage（卖点覆盖）
- Level 5: 所有卖点都有明确视觉+文字呈现
- Level 1: 无任何卖点被提及

### cross_modal（跨模态一致性）
- Level 5: 口播/花字与画面完全一致，信息互补
- Level 1: 严重不一致，文字与画面矛盾

### hook_strength（前 3 秒钩子）
- Level 5: 强视觉冲击 + 明确利益点，完美抓住注意力
- Level 1: 平淡开场，无任何吸引力

### marketing_logic（营销逻辑）
- Level 5: 完整的 痛点→方案→证据→CTA 链路
- Level 1: 逻辑混乱，无营销结构

### audience_match（受众匹配）
- Level 5: 风格、语言、场景完美契合目标受众
- Level 1: 完全不匹配目标受众

## 验收标准

- [ ] 5 个 .txt 文件存在于 video_eval/prompts/
- [ ] 每个模板包含完整的 5 级评分标准
- [ ] 每个模板包含 JSON 输出格式说明
- [ ] sellpoint_coverage 模板含 {product_title} 和 {selling_points} 占位符
- [ ] cross_modal 模板含条件 ASR/OCR 段落
- [ ] 所有模板的输出格式一致（level/reasoning/evidence/suggestion）
