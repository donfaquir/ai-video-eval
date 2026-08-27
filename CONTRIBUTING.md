# Contributing to video-eval

Thank you for your interest in contributing to video-eval. This guide explains how to get started.

## How to Contribute

- **Bug reports** — Open an issue with reproduction steps, expected vs actual behavior, and environment info
- **Feature requests** — Open an issue describing the use case and proposed solution
- **Code contributions** — Fork, branch, implement, test, and submit a PR
- **Documentation** — Improvements to docs, examples, and docstrings are always welcome
- **Plugins** — Build and publish third-party plugins (see [docs/plugin-development.md](docs/plugin-development.md))

## Development Setup

```bash
# Clone the repository
git clone https://github.com/anthropics/video-eval.git
cd video-eval

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode with all dependencies
pip install -e '.[all,dev]'

# Verify setup
video-eval --version
pytest --co -q  # list tests without running
```

### Prerequisites

- Python 3.11+
- ffmpeg installed and on PATH
- (Optional) CUDA or MPS GPU for testing GPU-dependent plugins

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check lint
ruff check .

# Auto-fix what's possible
ruff check --fix .

# Format
ruff format .
```

### Conventions

- All code, comments, docstrings, log messages, and error strings in **English**
- Type hints on all public functions
- Docstrings on all public classes and methods (Google style)
- No wildcard imports
- Keep modules focused: one plugin per file

## Testing

We use [pytest](https://docs.pytest.org/) with coverage tracking.

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=video_eval --cov-report=term-missing

# Run a specific test file
pytest tests/test_registry.py

# Run tests matching a pattern
pytest -k "test_pipeline"
```

### Writing Tests

- Place tests in `tests/` mirroring the source structure
- Use fixtures for common setup (DeviceManager, config dicts)
- Test both success paths and error/degradation paths
- Mock external dependencies (API calls, GPU operations)
- Target >80% coverage for new code; >90% for `core/` modules

Example test structure:

```python
import pytest
from video_eval.core.device import DeviceManager


@pytest.fixture
def device_manager():
    return DeviceManager()


def test_evaluator_scored(device_manager):
    """Evaluator produces valid score on normal input."""
    ...


def test_evaluator_missing_field(device_manager):
    """Evaluator returns skipped when required field missing."""
    ...
```

## Pull Request Process

1. **Branch from `main`** — Use a descriptive branch name: `feat/watermark-evaluator`, `fix/batch-exit-code`, `docs/config-reference`

2. **Keep PRs focused** — One logical change per PR. Split large features into smaller PRs when possible.

3. **Write tests** — All new functionality must have test coverage. Bug fixes should include a regression test.

4. **Update docs** — If your change affects user-facing behavior, update relevant documentation.

5. **Run CI locally** before pushing:
   ```bash
   ruff check .
   pytest --cov=video_eval
   ```

6. **Write a clear PR description** — Explain what the change does, why it's needed, and how to test it.

7. **Address review feedback** — Push new commits (don't force-push during review).

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add watermark detection evaluator
fix: batch mode exit code for mixed results
docs: add configuration reference
refactor: extract topo-sort into utility function
test: add pipeline integration tests
chore: update ruff to 0.5.0
```

## Project Structure

```
video-eval/
├── video_eval/           # Source code
│   ├── core/             # Framework internals (registry, pipeline, config, device)
│   ├── evaluators/       # Built-in evaluator plugins
│   ├── backends/         # VLM backend plugins
│   ├── extractors/       # Feature extractor plugins
│   ├── fusions/          # Score fusion plugins
│   └── prompts/          # VLM prompt templates
├── tests/                # Test suite
├── docs/                 # Documentation
├── config.yaml.example   # Example configuration
└── pyproject.toml        # Project metadata and dependencies
```

## License

By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
