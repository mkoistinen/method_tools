"""
Using ``functools.lru_cache`` on methods causes memory leaks.

The class-level cache holds strong references to ``self``, preventing
instances from being garbage collected. This module provides
``lru_method_cache``, a drop-in replacement for ``functools.lru_cache``
that stores the cache on each instance so it is freed when the instance
is collected.

Unlike ``functools.lru_cache``, this implementation normalizes call
signatures so that positional and keyword forms of the same argument
produce the same cache key. For example, ``obj.foo(1, 2)`` and
``obj.foo(1, b=2)`` will share a single cache entry.
"""
import functools
import inspect
import warnings
import weakref
from collections import OrderedDict
from threading import Lock
from typing import Any, NamedTuple


class _CacheInfo(NamedTuple):
    """Statistics for a method cache instance."""

    hits: int
    misses: int
    max_size: int | None
    cur_size: int


class _MethodCacheDescriptor:
    """Provide per-instance LRU caching for methods.

    Each instance gets its own cache (stored on the descriptor keyed by
    instance id), so the cache lifetime is tied to the instance — no
    memory leak from a class-level cache holding strong references to
    ``self``.

    If you are unfamiliar with Descriptors please see here:
    https://docs.python.org/3/howto/descriptor.html

    Parameters
    ----------
    method : callable
        The method to cache.
    max_size : int or None
        Maximum number of cached results. ``None`` means unlimited.
        Defaults to 128, matching ``functools.lru_cache``.
    typed : bool
        If True, arguments of different types are cached separately.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        method: Any,
        *,
        max_size: int | None = 128,
        typed: bool = False,
    ) -> None:
        self._method = method
        self._max_size = max_size
        self._typed = typed
        self._caches: dict[int, OrderedDict[Any, Any]] = {}
        self._locks: dict[int, Lock] = {}
        self._wrappers: dict[int, Any] = {}
        self._hits: dict[int, int] = {}
        self._misses: dict[int, int] = {}
        self._name: str = method.__name__
        self._param_names: tuple[str, ...] = tuple(
            list(inspect.signature(method).parameters)[1:]
        )
        functools.update_wrapper(self, method)  # type: ignore[arg-type]

    def _make_key(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, ...]:
        # Normalize: map positional args to their parameter names,
        # merge with kwargs, and produce a canonical sorted-items key.
        # strict=False: callers may pass fewer positional args than
        # parameters, with the remainder supplied as kwargs.
        normalized = dict(zip(self._param_names, args, strict=False))
        normalized.update(kwargs)
        key = tuple(sorted(normalized.items()))
        if self._typed:
            key += tuple(
                (k, type(v)) for k, v in sorted(normalized.items())
            )
        return key

    def _cleanup(self, obj_id: int) -> None:
        """Remove cache data for a garbage-collected instance."""
        self._caches.pop(obj_id, None)
        self._locks.pop(obj_id, None)
        self._hits.pop(obj_id, None)
        self._misses.pop(obj_id, None)
        self._wrappers.pop(obj_id, None)

    def __set_name__(self, _owner: type, name: str) -> None:
        self._name = name

    def __get__(self, obj: Any, _objtype: type | None = None) -> Any:
        if obj is None:
            return self

        obj_id = id(obj)
        if obj_id not in self._caches:
            self._caches[obj_id] = OrderedDict()
            self._locks[obj_id] = Lock()
            self._hits[obj_id] = 0
            self._misses[obj_id] = 0
            try:
                weakref.finalize(obj, self._cleanup, obj_id)
            except TypeError:
                warnings.warn(
                    f"{type(obj).__name__} does not support weak"
                    f" references; cached results for"
                    f" {self._name!r} will not be cleaned up"
                    f" automatically when the instance is deleted",
                    stacklevel=2,
                )

        cache = self._caches[obj_id]
        lock = self._locks[obj_id]
        max_size = self._max_size
        hits_ref = self._hits
        misses_ref = self._misses
        method = self._method

        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = self._make_key(args, kwargs)
            with lock:
                if key in cache:
                    cache.move_to_end(key)
                    hits_ref[obj_id] += 1
                    return cache[key]
            result = method(obj, *args, **kwargs)
            with lock:
                cache[key] = result
                misses_ref[obj_id] += 1
                if max_size is not None and len(cache) > max_size:
                    cache.popitem(last=False)
            return result

        def cache_info() -> _CacheInfo:
            with lock:
                return _CacheInfo(
                    hits_ref.get(obj_id, 0),
                    misses_ref.get(obj_id, 0),
                    max_size,
                    len(cache),
                )

        def cache_clear() -> None:
            with lock:
                cache.clear()
                hits_ref[obj_id] = 0
                misses_ref[obj_id] = 0

        wrapper.cache_info = cache_info  # type: ignore[attr-defined]
        wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]

        return wrapper


def lru_method_cache(
    method: Any = None,
    *,
    max_size: int | None = 128,
    typed: bool = False,
) -> Any:
    """
    Replace ``functools.lru_cache`` for use on methods.

    Store cached results per-instance so the cache lifetime is tied to
    the instance — no memory leak from a class-level cache holding
    strong references to ``self``.

    Improve on ``functools.lru_cache`` by normalizing call signatures so
    that positional and keyword forms of the same argument produce the
    same cache key.

    Support the same interface as ``functools.lru_cache``:

    - ``max_size``: maximum cache entries (default 128, ``None`` for
      unlimited)
    - ``typed``: cache arguments of different types separately
    - ``cache_info()``: returns a named tuple of hits, misses,
      max_size, cur_size
    - ``cache_clear()``: clears the cache and resets statistics

    Can be used with or without parentheses::

        @lru_method_cache
        def get_value(self, key): ...

        @lru_method_cache(max_size=256, typed=True)
        def get_value(self, key): ...
    """
    def _wrap(m: Any) -> _MethodCacheDescriptor:
        if isinstance(m, (classmethod, staticmethod)):
            kind = type(m).__name__
            msg = (
                f"@lru_method_cache cannot wrap a {kind}. "
                f"Use functools.lru_cache for {kind} methods (they do not "
                f"cause the memory leak that lru_method_cache is designed "
                f"to prevent)."
            )
            raise TypeError(msg)
        return _MethodCacheDescriptor(m, max_size=max_size, typed=typed)

    if method is not None:
        return _wrap(method)
    return _wrap
