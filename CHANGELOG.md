# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-XX-XX

### Added

- Core framework: Registry, Pipeline, DeviceManager, ConfigLoader
- 5 evaluators: technical_quality, compliance, aigc_defect, product_fidelity, vlm_judge
- 4 extractors: video_meta, asr, ocr, clip_features
- 3 backends: mock, api (Gemini/OpenAI), local (Qwen3-VL)
- WeightedVetoFusion strategy
- CLI: eval, extract, batch, plugins, config check, device info
- Plugin system with entry-points support
- Three-layer configuration (defaults < YAML file < CLI --set overrides)
- Batch processing with directory scan and CSV manifest
- Cross-device support: CUDA / Apple MPS / CPU auto-detection
- Structured JSON/JSONL output with evidence and suggestions
- Exit codes for CI integration (0=pass, 1=grade-C, 3=reject, 4=error, 6=extraction-failure)
