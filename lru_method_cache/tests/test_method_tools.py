import gc
from typing import TYPE_CHECKING, Any

import pytest

from lru_method_cache.lru_method_cache import (
    _MethodCacheDescriptor,  # pyright: ignore[reportPrivateUsage]
    lru_method_cache,
)


if TYPE_CHECKING:
    from lru_method_cache.lru_method_cache import (
        _CacheInfo,  # pyright: ignore[reportPrivateUsage]
    )


class Counter:
    """Helper class for testing lru_method_cache."""

    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        self.call_count: int = 0

    @lru_method_cache
    def compute(self, x: int) -> int:
        """Return x doubled, tracking call count."""
        self.call_count += 1
        return x * 2

    @lru_method_cache(max_size=None)
    def unbounded(self, x: int) -> int:
        """Cache with no size limit."""
        self.call_count += 1
        return x * 3

    @lru_method_cache(max_size=2)
    def small_cache(self, x: int) -> int:
        """Cache limited to 2 entries."""
        self.call_count += 1
        return x + 1

    @lru_method_cache(typed=True)
    def typed_compute(self, x: float) -> float:
        """Cache that distinguishes argument types."""
        self.call_count += 1
        return x

    @lru_method_cache(max_size=128)
    def with_kwargs(self, x: int, *, multiplier: int = 1) -> int:
        """Cache with keyword arguments."""
        self.call_count += 1
        return x * multiplier

    @lru_method_cache
    def mixed_args(self, a: int, b: int, c: int = 0) -> int:
        """Cache with args that can be passed positionally or as kwargs."""
        self.call_count += 1
        return a + b + c


class TestBasicCaching:
    """Test that results are cached and returned correctly."""

    def test_returns_correct_result(self) -> None:
        """Ensure the right results are returned."""
        c: Counter = Counter()
        assert c.compute(5) == 10

    def test_caches_repeated_calls(self) -> None:
        """Ensure caching is actually occurring."""
        c: Counter = Counter()
        c.compute(5)
        c.compute(5)
        c.compute(5)
        assert c.call_count == 1

    def test_different_args_cached_separately(self) -> None:
        """Test that the cache keeps uniquely arg'ed calls separate."""
        c: Counter = Counter()
        assert c.compute(5) == 10
        assert c.compute(6) == 12
        assert c.call_count == 2

    def test_decorator_without_parentheses(self) -> None:
        """Test the non-parenthesized form."""
        c: Counter = Counter()
        c.compute(1)
        c.compute(1)
        assert c.call_count == 1

    def test_decorator_with_parentheses(self) -> None:
        """Test the parenthesized form."""
        c: Counter = Counter()
        c.unbounded(1)
        c.unbounded(1)
        assert c.call_count == 1
        assert c.unbounded(1) == 3


class TestPerInstanceCache:
    """Test that caches are independent per instance."""

    def test_separate_instances_have_separate_caches(self) -> None:
        """Test that the cache keeps uniquely arg'ed calls separate."""
        a: Counter = Counter()
        b: Counter = Counter()
        a.compute(5)
        b.compute(5)
        assert a.call_count == 1
        assert b.call_count == 1

    def test_cache_info_independent_per_instance(self) -> None:
        """Test that the cached returns are stored on the instance."""
        a: Counter = Counter()
        b: Counter = Counter()
        a.compute(1)
        a.compute(2)
        b.compute(1)
        assert a.compute.cache_info().cur_size == 2
        assert b.compute.cache_info().cur_size == 1


class TestCacheInfo:
    """Test cache_info() returns correct statistics."""

    def test_initial_cache_info(self) -> None:
        """Test that statistics are correctly computed."""
        c: Counter = Counter()
        info: _CacheInfo = c.compute.cache_info()
        assert info.hits == 0
        assert info.misses == 0
        assert info.max_size == 128
        assert info.cur_size == 0

    def test_hits_and_misses(self) -> None:
        """Test that hit/miss info is captured correctly."""
        c: Counter = Counter()
        c.compute(1)
        c.compute(2)
        c.compute(1)
        info: _CacheInfo = c.compute.cache_info()
        assert info.hits == 1
        assert info.misses == 2
        assert info.cur_size == 2

    def test_cache_info_is_named_tuple(self) -> None:
        """Test that the cache is, in-fact maintained as a collection of NamedTuples."""
        c: Counter = Counter()
        c.compute(1)
        info: _CacheInfo = c.compute.cache_info()
        hits: int
        misses: int
        max_size: int | None
        cur_size: int
        hits, misses, max_size, cur_size = info
        assert hits == 0
        assert misses == 1
        assert max_size == 128
        assert cur_size == 1

    def test_unbounded_max_size(self) -> None:
        """Test that we can support un-bounded usage."""
        c: Counter = Counter()
        assert c.unbounded.cache_info().max_size is None


class TestCacheClear:
    """Test cache_clear() resets cache and statistics."""

    def test_clears_cached_values(self) -> None:
        """Test cache_clear does its job."""
        c: Counter = Counter()
        c.compute(1)
        c.compute.cache_clear()
        c.compute(1)
        assert c.call_count == 2

    def test_resets_statistics(self) -> None:
        """Test that clear_cache correctly resets statistics."""
        c: Counter = Counter()
        c.compute(1)
        c.compute(1)
        c.compute.cache_clear()
        info: _CacheInfo = c.compute.cache_info()
        assert info.hits == 0
        assert info.misses == 0
        assert info.cur_size == 0


class TestLRUEviction:
    """Test that LRU eviction works with max_size."""

    def test_evicts_least_recently_used(self) -> None:
        """Test that only the most recently used items are kept."""
        c: Counter = Counter()
        c.small_cache(1)
        c.small_cache(2)
        c.small_cache(3)  # should evict 1
        assert c.small_cache.cache_info().cur_size == 2
        # 1 was evicted, calling it again should be a miss
        c.call_count = 0
        c.small_cache(1)
        assert c.call_count == 1

    def test_access_refreshes_lru_order(self) -> None:
        """Test that accessing a cache hit makes the item more recent."""
        c: Counter = Counter()
        c.small_cache(1)
        c.small_cache(2)
        c.small_cache(1)  # refresh 1, so 2 is now LRU
        c.small_cache(3)  # should evict 2, not 1
        c.call_count = 0
        c.small_cache(1)  # should be a hit
        assert c.call_count == 0
        c.small_cache(2)  # should be a miss (was evicted)
        assert c.call_count == 1


class TestTypedCaching:
    """Test typed=True caches different types separately."""

    def test_int_and_float_cached_separately(self) -> None:
        """Ensure that `typed` works as expected when types differ."""
        c: Counter = Counter()
        c.typed_compute(1)
        c.typed_compute(1.0)
        assert c.call_count == 2

    def test_same_type_still_cached(self) -> None:
        """Ensure that `typed` works as expected when same type."""
        c: Counter = Counter()
        c.typed_compute(1)
        c.typed_compute(1)
        assert c.call_count == 1


class TestKwargCaching:
    """Test caching with keyword arguments."""

    def test_different_kwargs_cached_separately(self) -> None:
        """Ensure kwargs are respected."""
        c: Counter = Counter()
        c.with_kwargs(5, multiplier=2)
        c.with_kwargs(5, multiplier=3)
        assert c.call_count == 2

    def test_same_kwargs_cached(self) -> None:
        """Ensure kwargs are respected."""
        c: Counter = Counter()
        c.with_kwargs(5, multiplier=2)
        c.with_kwargs(5, multiplier=2)
        assert c.call_count == 1
        assert c.with_kwargs(5, multiplier=2) == 10


class TestSignatureNormalization:
    """Test that positional and keyword forms share a cache entry."""

    def test_positional_and_keyword_share_cache(self) -> None:
        """Ensure foo(1, 2) and foo(1, b=2) hit the same entry."""
        c: Counter = Counter()
        assert c.mixed_args(1, 2) == 3
        assert c.mixed_args(1, b=2) == 3
        assert c.call_count == 1

    def test_all_keyword_shares_cache(self) -> None:
        """Ensure foo(a=1, b=2) also hits the same entry."""
        c: Counter = Counter()
        c.mixed_args(1, 2)
        c.mixed_args(a=1, b=2)
        assert c.call_count == 1

    def test_different_values_still_separate(self) -> None:
        """Ensure normalization doesn't collapse different values."""
        c: Counter = Counter()
        c.mixed_args(1, 2)
        c.mixed_args(1, 3)
        assert c.call_count == 2

    def test_optional_arg_positional_vs_keyword(self) -> None:
        """Ensure foo(1, 2, 3) and foo(1, 2, c=3) share a cache entry."""
        c: Counter = Counter()
        assert c.mixed_args(1, 2, 3) == 6
        assert c.mixed_args(1, 2, c=3) == 6
        assert c.call_count == 1

    def test_omitted_default_differs_from_explicit(self) -> None:
        """Omitting a default and passing it explicitly are separate entries.

        This is expected: filling in defaults would require inspecting
        default values on every call, adding overhead for little benefit.
        """
        c: Counter = Counter()
        assert c.mixed_args(1, 2) == 3
        assert c.mixed_args(1, 2, c=0) == 3
        assert c.call_count == 2


class TestClassAccess:
    """Test descriptor behavior when accessed on the class."""

    def test_class_access_returns_descriptor(self) -> None:
        """Test that we can access the descriptor."""
        descriptor: Any = Counter.__dict__["compute"]
        assert isinstance(descriptor, _MethodCacheDescriptor)


class TestCleanup:
    """Test that cache data is cleaned up when instances are deleted."""

    def test_cleanup_on_garbage_collection(self) -> None:
        """Test that memory is not leaked."""
        c: Counter = Counter()
        c.compute(1)
        descriptor: _MethodCacheDescriptor = Counter.__dict__["compute"]
        obj_id: int = id(c)
        assert obj_id in descriptor._caches  # pyright: ignore[reportPrivateUsage]
        del c
        gc.collect()
        assert obj_id not in descriptor._caches  # pyright: ignore[reportPrivateUsage]


class TestClassmethodStaticmethodRejection:
    """Test that classmethod and staticmethod are rejected."""

    def test_rejects_classmethod(self) -> None:
        """Test that a TypeError is raised on a classmethod."""
        with pytest.raises(TypeError, match="classmethod"):

            class _Bad:  # pyright: ignore[reportUnusedClass]
                @lru_method_cache
                @classmethod
                def foo(cls) -> None:
                    ...

    def test_rejects_staticmethod(self) -> None:
        """Test that a TypeError is raised on a staticmethod."""
        with pytest.raises(TypeError, match="staticmethod"):

            class _Bad:  # pyright: ignore[reportUnusedClass]
                @lru_method_cache
                @staticmethod
                def foo() -> None:
                    ...

    def test_rejects_classmethod_with_args(self) -> None:
        """Test that a TypeError is raised on a classmethod with args."""
        with pytest.raises(TypeError, match="classmethod"):

            class _Bad:  # pyright: ignore[reportUnusedClass]
                @lru_method_cache(max_size=64)
                @classmethod
                def foo(cls) -> None:
                    ...


class TestWeakrefWarning:
    """Test that a warning is issued for non-weak-referenceable objects."""

    def test_warns_on_non_weakrefable_instance(self) -> None:
        """Test that warnings are raised if the class instance uses slots."""
        class NoWeakref:
            __slots__ = ("call_count",)

            def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
                self.call_count: int = 0

            @lru_method_cache
            def compute(self, x: int) -> int:
                self.call_count += 1
                return x * 2

        obj: NoWeakref = NoWeakref()
        with pytest.warns(UserWarning, match="does not support weak"):
            obj.compute(1)

        # Caching still works despite the warning
        obj.compute(1)
        assert obj.call_count == 1
