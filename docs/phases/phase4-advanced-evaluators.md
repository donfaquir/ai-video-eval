# Phase 4：高级评估器

## 目标

实现 AIGCDefect 和 ProductFidelity 两个 GPU 评估器，完成全部内置评估维度。

## 前置依赖

Phase 2 完成（clip_features 抽取器就绪）。可与 Phase 3 并行开发。

## 交付范围

### 1. AIGCDefect 评估器（详设 §6.4）

```python
@register_evaluator("aigc_defect")
class AIGCDefectEvaluator(BaseEvaluator):
    name = "aigc_defect"
    device_requirement = "gpu"
    requires = ["frames"]
    default_weights = None     # 由 config weights 表声明（main_image: 0.15, external: 0.10）
    config_schema = {
        "model": {"type": "str", "default": "openai/clip-vit-large-patch14"},
        "defect_threshold": {"type": "float", "default": 0.6},
    }
```

实现要点：
- `__enter__`：加载 CLIP-ViT-L/14 模型（独立于 clip_features 的 SigLIP）
- `evaluate(context)`：对 frames 做 zero-shot 分类，检测"扭曲/变形/不自然"类瑕疵
- 输出：score=[0,1]（1=无瑕疵），evidence=被标记帧列表（frame_idx + timestamp + defect_prob）
- 显存：约 1GB（resident 预算表已计入）
- MPS 兼容

### 2. ProductFidelity 评估器（详设 §6.5）

```python
@register_evaluator("product_fidelity")
class ProductFidelityEvaluator(BaseEvaluator):
    name = "product_fidelity"
    device_requirement = "gpu"
    requires = ["frames", "clip_features", "product_info"]
    default_weights = None     # 由 config weights 表声明
```

实现要点：
- 消费 clip_features（SigLIP tensor，由 clip_features 抽取器产出）
- 消费 product_info.main_image_paths（商家主图）
- 计算视频帧特征与主图特征的余弦相似度
- 输出：score=[0,1]，evidence=每个卖点的最佳匹配帧+相似度
- product_info 缺失时由 F3 自动跳过（requires 包含 product_info）
- MPS 兼容

### 3. 双模型显存管理

- aigc_defect 自带 CLIP-L（约 1GB）与 clip_features 的 SigLIP（约 2GB）是两份独立模型
- resident 模式显存预算：Qwen3-VL 16GB + SigLIP 2GB + Whisper 3GB + CLIP-L 1GB = 22GB
- sequential 模式：评估器逐个加载释放，显存峰值 = max(单评估器)
- can_load_model 校验：resident 下预估峰值超预算 → 退出码 2

### 4. 测试

| 测试 | 覆盖点 |
|------|--------|
| `test_aigc_defect.py` | CLIP 加载/zero-shot 分类/阈值过滤/evidence 格式/GPU↔CPU 跳过 |
| `test_product_fidelity.py` | SigLIP 特征消费/主图相似度计算/product_info 缺失 F3 跳过 |

## 验收标准

- [ ] GPU 机器：`video-eval eval --video test.mp4 --video-type main_image --product-title "..." --product-images img.jpg` 输出 aigc_defect + product_fidelity 得分
- [ ] CPU 机器：两个评估器被 F2 device 过滤跳过，输出 `skipped: device_unavailable`
- [ ] product_info 缺失时：product_fidelity 被 F3 跳过（missing_product_info）
- [ ] resident 模式显存预估超限时：退出码 2 并提示改 sequential
- [ ] 全部内置评估器就绪后，`video-eval plugins` 列出完整 5 个评估器 + 4 个抽取器 + 3 个后端

## 参考详设章节

§6.4（AIGCDefect）、§6.5（ProductFidelity）、§4.1（显存预算）、§4.6（异常表 C4 行）
