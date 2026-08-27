# Phase 5：开源发布

## 目标

完成文档、Docker、CI/CD，发布 v0.1.0。项目达到可公开开源的质量标准。

## 前置依赖

Phase 1-4 全部完成。

## 交付范围

### 1. Docker

| 文件 | 说明 |
|------|------|
| `docker/Dockerfile.cpu` | python:3.11-slim 基础，CPU PyTorch，VLM 默认走 API |
| `docker/Dockerfile.gpu` | nvidia/cuda:12.1 基础，CUDA PyTorch，VLM 走 local |
| `docker-compose.yml` | 开箱即用，挂载 data/config/results 卷 |

验收：
- `docker compose up` CPU 版能跑 mock 后端端到端
- GPU 版能加载 Qwen3-VL 并完成 local 后端评估

### 2. 文档

| 文件 | 内容 |
|------|------|
| `README.md` | 项目介绍/快速开始/安装/基本用法/配置/插件开发入口 |
| `docs/getting-started.md` | 详细安装指南（pip/docker/从源码）|
| `docs/configuration.md` | 配置参考（从 §7.3 生成）|
| `docs/plugin-development.md` | 插件开发指南（评估器/后端/抽取器/融合策略，含示例）|
| `docs/architecture.md` | 架构概览（面向贡献者）|
| `CHANGELOG.md` | v0.1.0 变更日志 |
| `CONTRIBUTING.md` | 贡献指南 + code of conduct |
| `LICENSE` | Apache-2.0 |

### 3. CI/CD（GitHub Actions）

| Workflow | 触发 | 内容 |
|---------|------|------|
| `ci.yml` | push/PR | lint(ruff) + type-check(mypy) + test(pytest) + coverage |
| `docker.yml` | tag push | 构建并推送 Docker 镜像 |
| `release.yml` | tag push | 构建 wheel + 发布 PyPI |

### 4. Examples

```
examples/
├── basic_eval.py           # 最简单的单视频评估
├── batch_eval.py           # 批量评估
├── custom_evaluator.py     # 自定义评估器插件
├── custom_backend.py       # 自定义 VLM 后端
└── config.yaml             # 示例配置（带注释）
```

### 5. 项目元数据

- pyproject.toml 完善：classifiers / urls / optional-deps 分组（`[asr]` / `[gpu]` / `[dev]`）
- entry-points 声明（video_eval.evaluators / .backends / .extractors / .fusions）
- `py.typed` marker

### 6. 质量门槛

- [ ] ruff lint 零 warning
- [ ] mypy strict 模式通过（或 basic 模式零 error）
- [ ] pytest 覆盖率 > 80%（core 模块 > 90%）
- [ ] 所有 public API 有 docstring
- [ ] README 中的所有代码示例可运行

## 验收标准

- [ ] `pip install video-eval` 能在干净 venv 中安装并运行 `video-eval --help`
- [ ] `pip install 'video-eval[asr]'` 安装 ASR 依赖后 asr 抽取器可用
- [ ] Docker CPU 版构建 < 5min，镜像 < 2GB
- [ ] GitHub Actions CI 绿色通过
- [ ] README 快速开始步骤可照搬执行
- [ ] v0.1.0 tag + PyPI 发布成功

## 参考详设章节

§9（插件开发 API）、§1.9（插件发现/entry-points）、design.md §12（Docker）、design.md §13.4（Phase 4 开源准备）
