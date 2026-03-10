# lru-method-cache

[![CI](https://github.com/mkoistinen/lru-method-cache/actions/workflows/ci.yml/badge.svg)](https://github.com/mkoistinen/lru-method-cache/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/mkoistinen/lru-method-cache/branch/main/graph/badge.svg)](https://codecov.io/gh/mkoistinen/lru-method-cache)
[![PyPI](https://img.shields.io/pypi/v/lru-method-cache)](https://pypi.org/project/lru-method-cache/)
[![Python](https://img.shields.io/pypi/pyversions/lru-method-cache)](https://pypi.org/project/lru-method-cache/)

A drop-in replacement for `functools.lru_cache` designed for methods.

## The problem

Using `functools.lru_cache` on methods causes memory leaks. The cache is
stored on the class (via the decorator), and every cached call stores a
strong reference to `self` as part of the cache key. This prevents
instances from being garbage collected, even after all other references
are gone:

```python
from functools import lru_cache

class MyClass:
    @lru_cache(maxsize=128)
    def expensive(self, x):
        return x * 2

a = MyClass()
a.expensive(1)
del a  # instance is NOT garbage collected — the cache still holds a reference
```

This is a well-known issue with no satisfying solution in the standard
library. Here are some references to learn more.

- Python Docs: https://docs.python.org/3/faq/programming.html#how-do-i-cache-method-calls
- Blog post: https://rednafi.com/python/lru_cache_on_methods/
- Stack Overflow: https://stackoverflow.com/q/33672412

## The solution

`lru-method-cache` provides `lru_method_cache`, which stores the cache
**per instance** using a descriptor. When an instance is garbage
collected, its cache is automatically cleaned up via `weakref.finalize`.

```python
from lru_method_cache import lru_method_cache

class MyClass:
    @lru_method_cache(maxsize=128)
    def expensive(self, x):
        return x * 2

a = MyClass()
a.expensive(1)
del a  # instance is garbage collected, the respective cache is freed
```

### Signature normalization

Unlike `functools.lru_cache`, `lru_method_cache` normalizes call
signatures so that positional and keyword forms of the same argument
produce the same cache key:

```python
class MyClass:
    @lru_method_cache
    def compute(self, a, b, c=0):
        return a + b + c

obj = MyClass()
obj.compute(1, 2)       # miss
obj.compute(1, b=2)     # hit — same as above
obj.compute(a=1, b=2)   # hit — same as above
```

`functools.lru_cache` treats these as three separate cache entries.

## Usage

```python
from lru_method_cache import lru_method_cache

class MyClass:
    # Without parentheses (defaults: max_size=128, typed=False)
    @lru_method_cache
    def method_a(self, x):
        ...

    # With parentheses
    @lru_method_cache(max_size=256, typed=True)
    def method_b(self, x):
        ...

    # Unlimited cache
    @lru_method_cache(max_size=None)
    def method_c(self, x):
        ...
```

The familiar `cache_info()` and `cache_clear()` methods are available on
each bound method, just like `functools.lru_cache`:

```python
obj = MyClass()
obj.method_a(1)
obj.method_a(2)
obj.method_a(1)  # cache hit

obj.method_a.cache_info()
# CacheInfo(hits=1, misses=2, max_size=128, cur_size=2)

obj.method_a.cache_clear()
```

### Parameters

| Parameter  | Default | Description |
|------------|---------|-------------|
| `max_size` | `128`   | Maximum number of cached results. `None` for unlimited. |
| `typed`    | `False` | If `True`, arguments of different types are cached separately (e.g., `1` and `1.0` are distinct). |

### Classmethods and staticmethods

`lru_method_cache` is only for instance methods. Applying it to a
`classmethod` or `staticmethod` raises `TypeError` with a helpful
message — those don't have the memory leak problem, so
`functools.lru_cache` works fine for them.

### Classes with `__slots__`

If a class uses `__slots__` without `__weakref__`, the cache still works
but a `UserWarning` is issued because cleanup cannot happen automatically
when the instance is deleted. Add `"__weakref__"` to `__slots__` to
enable automatic cleanup.

## How it compares

There are certain other clever solutions for the issue of using `lru_cache` on class methods. Here is a comparison of this project to some of other solutions:

| Approach | Memory-safe | Signature normalization | Per-instance cache | Thread-safe |
|----------|:-----------:|:-----------------------:|:------------------:|:-----------:|
| `functools.lru_cache` | No | No | No | Yes |
| [`methodtools`](https://github.com/clee704/methodtools)`.lru_cache` | Yes | No | Yes | Yes |
| [`cachetools`](https://github.com/tkem/cachetools) + manual wiring | Yes | No | Manual | Manual |
| **[`lru-method-cache`](https://github.com/mkoistinen/lru-method-cache)`.lru_method_cache`** | **Yes** | **Yes** | **Yes** | **Yes** |

## Design philosophy

`lru-method-cache` uses only Python's standard library — no third-party
dependencies. The implementation is a single, short module that is easy
to read, inspect, and audit. If you want to understand exactly what your
caching decorator does, you can review the entire source in a few
moments.

## Installation

```bash
pip install lru-method-cache
```

## Requirements

Python 3.10+. No third-party dependencies.

## License

MIT
