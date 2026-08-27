# Spec 01：项目脚手架

## 目标

创建项目目录结构、pyproject.toml、所有 `__init__.py` 占位，使项目可 `pip install -e .` 并执行 `video-eval --help`（输出帮助文本即可，命令实现在后续 spec 中补充）。

## 依赖

无（首个 spec）。

## 产出文件

```
video-eval/
├── pyproject.toml
├── video_eval/
│   ├── __init__.py             # __version__ = "0.1.0"
│   ├── cli.py                  # click group 骨架（仅 --help 能跑）
│   ├── core/
│   │   ├── __init__.py
│   │   ├── registry.py         # 占位
│   │   ├── base.py             # 占位
│   │   ├── schemas.py          # 占位
│   │   ├── device.py           # 占位
│   │   ├── pipeline.py         # 占位
│   │   ├── config.py           # 占位
│   │   └── exceptions.py       # 占位
│   ├── evaluators/
│   │   └── __init__.py
│   ├── extractors/
│   │   └── __init__.py
│   ├── backends/
│   │   └── __init__.py
│   ├── fusions/
│   │   └── __init__.py
│   └── prompts/                # 空目录（Phase 3）
│       └── .gitkeep
├── tests/
│   ├── __init__.py
│   └── conftest.py             # pytest fixtures 骨架
└── config.yaml.example         # 完整默认配置（从详设 §7.3 抄）
```

## pyproject.toml 关键内容

```toml
[project]
name = "video-eval"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "click>=8.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "tqdm>=4.0",
]

[project.optional-dependencies]
asr = ["faster-whisper>=1.0"]
gpu = ["torch>=2.0", "open-clip-torch>=2.20"]
dev = ["pytest>=7.0", "pytest-cov", "ruff"]

[project.scripts]
video-eval = "video_eval.cli:main"

[project.entry-points."video_eval.evaluators"]
[project.entry-points."video_eval.backends"]
[project.entry-points."video_eval.extractors"]
[project.entry-points."video_eval.fusions"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## cli.py 骨架

```python
import click

@click.group()
@click.version_option()
def main():
    """video-eval: AI-generated video quality evaluation framework."""
    pass

@main.command()
def eval():
    """Evaluate a single video."""
    click.echo("eval: not implemented yet")

@main.command()
def batch():
    """Batch evaluate videos."""
    click.echo("batch: not implemented yet")

@main.command()
def plugins():
    """List registered plugins."""
    click.echo("plugins: not implemented yet")

@main.group()
def config():
    """Configuration management."""
    pass

@config.command()
def check():
    """Validate configuration."""
    click.echo("config check: not implemented yet")

@main.group()
def device():
    """Device information."""
    pass

@device.command()
def info():
    """Show device info."""
    click.echo("device info: not implemented yet")
```

## 验收标准

- [ ] `pip install -e .` 成功（无报错）
- [ ] `video-eval --help` 输出命令列表
- [ ] `video-eval --version` 输出 `0.1.0`
- [ ] `python -c "from video_eval.core import registry, base, schemas"` 不报错（空模块）
- [ ] `pytest --collect-only` 能发现 tests/ 目录

## 实现顺序

本 spec → spec02（数据模型）→ spec03（DeviceManager）→ ...
