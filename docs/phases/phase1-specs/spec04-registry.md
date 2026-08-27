# Spec 04：Registry 系统

## 目标

实现完整的 `Registry[T]` 泛型类 + `Placeholder` + 4 个全局实例 + 装饰器 API + 插件发现机制。这是整个插件系统的基座。

## 依赖

Spec 02（PluginMeta, 异常类）。

## 产出文件

- `video_eval/core/registry.py`

## Registry[T] 类

参考详设 §1.1。

```python
import threading
from typing import TypeVar, Generic, Callable

T = TypeVar("T")

class Placeholder(Generic[T]):
    def __init__(self, name: str, loader: Callable[[], type[T]], *, origin: str, module_path: str | None = None):
        self.name = name
        self.loader = loader
        self.origin = origin
        self.module_path = module_path

class Registry(Generic[T]):
    def __init__(self, name: str, base_cls: type[T], *, allow_override: bool = False):
        self.name = name
        self.base_cls = base_cls
        self.allow_override = allow_override
        self._entries: dict[str, type[T] | Placeholder[T]] = {}
        self._meta: dict[str, PluginMeta] = {}
        self._origins: dict[str, str] = {}
        self._broken: set[str] = set()
        self._frozen: bool = False
        self._lock = threading.RLock()
```

### 方法实现要求

| 方法 | 行为 |
|------|------|
| `register(name, *, cls=None, origin="explicit")` | 无 cls 时返回装饰器。提取类属性构造 PluginMeta → 按冲突表决策 → 持锁写入 `_entries`/`_meta`/`_origins`。冻结后抛 RegistryFrozenError |
| `get(name)` | 未找到 → NameNotFoundError（含 suggestions）；命中 Placeholder → 物化；命中 `_broken` → MaterializationError（不重试） |
| `get_meta(name)` | 直接返回 `_meta[name]`，不物化 |
| `has(name)` | name in `_entries` |
| `list()` | 返回所有注册名称列表（注册顺序） |
| `list_meta()` | 返回所有 PluginMeta 列表 |
| `freeze()` / `unfreeze()` | 设置/清除 `_frozen` |
| `origin(name)` | 返回来源标签 |
| `is_placeholder(name)` | isinstance check |
| `is_broken(name)` | name in `_broken` |

### 物化逻辑（8 步）

```
1. 检查 _broken → 直接抛 MaterializationError（附首次失败原因）
2. 获取 RLock
3. 双重检查（另一线程可能已完成物化）
4. 调用 loader()
5. 失败 → 加入 _broken, 抛 MaterializationError（记原始异常）
6. issubclass(loaded, base_cls) 校验 → 失败抛 MaterializationError
7. 替换 _entries[name] = loaded; 更新 _meta[name]（is_placeholder=False, 回填完整属性）
8. 释放锁
```

### 冲突检测表（详设 §1.7）

| 当前 | 新注册 | allow_override | 结果 |
|------|--------|----------------|------|
| Placeholder | Concrete | any | 允许升级 |
| Concrete | Concrete | True | 覆盖 |
| Concrete | Concrete | False | DuplicateRegistrationError |
| Concrete | Placeholder | any | DuplicateRegistrationError |
| Placeholder | Placeholder | any | 保留先注册者 |
| 无 | any | any | 正常注册 |

### 智能错误提示

```python
def _suggest_similar(self, name: str, max_results: int = 3, threshold: int = 3) -> list[str]:
    """Prefix match + substring match + Levenshtein distance, top N by score."""
    ...
```

NameNotFoundError 消息格式：
```
'tehcincial' not found in evaluator registry.
  Did you mean: 'technical_quality', 'aigc_defect'?
  Available: technical_quality, aigc_defect, ... (10 of 28, use list() for full list)
```

### PluginMeta 提取

装饰器注册时从类属性提取：

```python
def _extract_meta(cls: type, name: str, origin: str) -> PluginMeta:
    return PluginMeta(
        name=name,
        version=getattr(cls, "version", "0.1.0"),
        device_requirement=getattr(cls, "device_requirement", "any"),
        requires=getattr(cls, "requires", []),
        config_schema=getattr(cls, "config_schema", {}),
        provides=getattr(cls, "provides", []),
        criticality=getattr(cls, "criticality", "optional"),
        backend_config_key=getattr(cls, "backend_config_key", None),
        default_weights=getattr(cls, "default_weights", None),
        dimension_slots=getattr(cls, "dimension_slots", None),
        origin=origin,
        is_placeholder=False,
    )
```

### 装饰器 API

```python
def register_evaluator(name: str):
    def decorator(cls):
        if cls.name != name:
            raise RegistryError(f"Alias '{name}' != cls.name '{cls.name}'", "evaluator")
        evaluator_registry.register(name, cls=cls, origin="builtin")
        return cls
    return decorator

# 同理：register_backend, register_extractor, register_fusion
```

### 4 个全局实例

```python
from video_eval.core.base import BaseEvaluator, BaseBackend, BaseExtractor, BaseFusion

evaluator_registry: Registry[BaseEvaluator] = Registry("evaluator", BaseEvaluator)
extractor_registry: Registry[BaseExtractor] = Registry("extractor", BaseExtractor)
backend_registry: Registry[BaseBackend] = Registry("backend", BaseBackend)
fusion_registry: Registry[BaseFusion] = Registry("fusion", BaseFusion)
```

注意：由于循环导入，实际上 base_cls 校验在 register 时做 `issubclass(cls, self.base_cls)`。4 个实例的创建需放在 `registry.py` 底部或独立的 `_instances.py` 中，避免循环。

### 插件发现

```python
def scan_directory(registry: Registry, package_path: str) -> None:
    """Import all .py modules in directory, triggering decorators."""
    ...

def discover_entry_points(registry: Registry, group: str) -> None:
    """
    Discover entry_points from installed packages.
    Each EP → Placeholder (origin="entry-point").
    Errors per EP: catch DuplicateRegistrationError / load failure → warning, skip.
    """
    from importlib.metadata import entry_points
    ...
```

### 初始化函数

```python
def initialize_registries() -> None:
    """Called once at startup. Does NOT freeze."""
    scan_directory(evaluator_registry, "video_eval/evaluators")
    scan_directory(extractor_registry, "video_eval/extractors")
    scan_directory(backend_registry, "video_eval/backends")
    scan_directory(fusion_registry, "video_eval/fusions")
    discover_entry_points(evaluator_registry, "video_eval.evaluators")
    discover_entry_points(extractor_registry, "video_eval.extractors")
    discover_entry_points(backend_registry, "video_eval.backends")
    discover_entry_points(fusion_registry, "video_eval.fusions")
```

## 验收标准

- [ ] 基础注册/获取：`registry.register("foo", cls=Foo)` → `registry.get("foo") is Foo`
- [ ] 冲突表全组合测试通过（6 种情况）
- [ ] freeze 后 register 抛 RegistryFrozenError
- [ ] 物化失败进 `_broken`，二次 get 不重试直接抛
- [ ] NameNotFoundError 消息含智能提示
- [ ] 装饰器注册 + alias≠cls.name 时抛异常
- [ ] Placeholder 物化成功后 `is_placeholder()` 返回 False
- [ ] `list_meta()` 返回所有 PluginMeta（含未物化的宽松默认值）
- [ ] `discover_entry_points` 容错（单个 EP 失败不中断扫描）
- [ ] 线程安全：两个线程同时 get 同一 Placeholder 只物化一次
