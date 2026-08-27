# Spec 06：配置系统

## 目标

实现 `ConfigLoader` 类，支持三层优先级合并（CLI > config.yaml > 插件默认值）、`--set` 类型转换、完整配置校验。

## 依赖

Spec 02（ValidationIssue, ConfigError）、Spec 04（Registry 实例，用于校验）。

## 产出文件

- `video_eval/core/config.py`
- `config.yaml.example`（从详设 §7.3 生成）

## ConfigLoader 类

参考详设 §7.1-7.2。

```python
class ConfigLoader:
    def __init__(self, config_path: str | None = None):
        """
        Args:
            config_path: Path to config.yaml. None = ./config.yaml (with fallback to defaults).
        """
        self.config_path = config_path

    def load(self) -> dict:
        """
        Read config.yaml + merge plugin default values.
        - File not found → use all-defaults + print warning
        - YAML parse error → ConfigError
        """
        ...

    def merge_cli_overrides(self, config: dict, cli_args: list[str]) -> dict:
        """
        Parse --set KEY=VALUE args and deep-merge into config.
        Type conversion rules:
          1. If key exists in plugin config_schema → force schema type (fail → ConfigError)
          2. Else: heuristic (int → float → bool → comma-split list → str)
        Dotted key path = nested dict traversal.
        """
        ...

    def validate(self, config: dict) -> list[ValidationIssue]:
        """
        Full validation. Returns list of issues (error/warning).
        Errors cause exit code 2; warnings are displayed but don't block.
        """
        ...
```

### 三层合并逻辑

```python
def _merge_defaults(self, config: dict) -> dict:
    """
    For each registered evaluator/extractor/backend:
    - Read its config_schema
    - For each key with "default" value, set in config if not present
    - Do NOT override user-provided values
    """
    ...
```

### --set 类型转换

```python
def _convert_value(self, key_path: str, raw_value: str, config: dict) -> Any:
    """
    1. Find plugin owning this key path → check config_schema for type
    2. If found: force conversion (int/float/str/list/bool)
    3. If not found: heuristic guess
       - "123" → int
       - "1.5" → float
       - "true"/"false" → bool
       - "a,b,c" → ["a","b","c"]
       - else → str
    """
    ...
```

### 校验规则（全集）

按详设 §7.2 校验清单逐条实现：

| # | 校验 | 级别 | 说明 |
|---|------|------|------|
| 1 | 引用未注册评估器（enabled=true 但 registry 无） | error | 退出码 2 |
| 2 | required 字段缺失 | error | |
| 3 | 评估器段未知键（不在 config_schema） | warning | 容忍版本差异 |
| 4 | backends 段未知键 | warning | 按该 backend 自己的 schema |
| 5 | backends 下未注册后端名段 | warning | 仅被选中时升 error |
| 6 | strict_veto_dims 维度被 disabled/未注册 | error | 退出码 2（D3） |
| 7 | strict_veto_dims 维度依赖闭包不可达 | error | 退出码 2（D5/D9） |
| 8 | backend_config_key 指向未注册 backend | error | 退出码 2 |
| 9 | resident 模式预估显存超限 | error | 退出码 2，提示改 sequential |
| 10 | required 抽取器被禁用 | warning | 提示字段不可用 |
| 11 | 已启用无权重非 veto 维度 | warning | "装了等于没装" |
| 12 | config.yaml 不存在 | warning | 使用默认配置 |

### D9 Probe 集成

校验规则 #7 包含 D9 启动 probe：

```python
def _probe_strict_veto_extractors(self, config: dict, device_manager: DeviceManager) -> list[ValidationIssue]:
    """
    For each dim in strict_veto_dims:
      1. Find owning evaluator → get its requires
      2. Compute closure of extractors needed
      3. For each extractor in closure:
         - Instantiate → __enter__() → __exit__()
         - Failure → ValidationIssue(severity="error", message=...)
    Probe instances are NOT reused (v7 decision).
    """
    ...
```

### config_hash 计算

```python
def compute_config_hash(config: dict) -> str:
    """
    Canonical serialization (sorted keys) → SHA-256 → first 8 hex chars.
    Excludes runtime info (device detection, timestamps).
    """
    import hashlib, json
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]
```

## config.yaml.example

直接从详设 §7.3 的完整 YAML 复制，每个键附注释。

## 验收标准

- [ ] `ConfigLoader().load()` 在无 config.yaml 时返回全默认值 + warning
- [ ] `ConfigLoader("config.yaml.example").load()` 成功加载
- [ ] `merge_cli_overrides(cfg, ["evaluators.compliance.limit_words=最,第一"])` → list 类型
- [ ] `merge_cli_overrides(cfg, ["fusion.thresholds.A=0.8"])` → float 类型
- [ ] `validate()` 对默认配置返回 0 error（可能有 warning）
- [ ] 引用未注册评估器 → severity="error" issue
- [ ] strict_veto 维度被 disabled → severity="error" issue
- [ ] `compute_config_hash` 对相同语义配置返回相同 hash
- [ ] --set 类型转换失败 → ConfigError
