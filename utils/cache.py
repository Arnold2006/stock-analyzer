"""Disk-based cache utility using diskcache.

Provides a simple :func:`cached` decorator and a :class:`Cache` wrapper
that other modules can share.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
from typing import Any, Callable, TypeVar

import diskcache

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache")
_cache: diskcache.Cache | None = None


def get_cache() -> diskcache.Cache:
    """Return the shared :class:`diskcache.Cache` instance, creating it lazily.

    Returns
    -------
    diskcache.Cache
        The application-wide disk cache.
    """
    global _cache
    if _cache is None:
        _cache = diskcache.Cache(_CACHE_DIR, size_limit=2**30)
        logger.debug("Cache opened at %s", _CACHE_DIR)
    return _cache


F = TypeVar("F", bound=Callable[..., Any])


def cached(ttl: int = 300) -> Callable[[F], F]:
    """Decorator that caches the return value of a function on disk.

    Parameters
    ----------
    ttl:
        Time-to-live in seconds (default: 300 s = 5 min).

    Returns
    -------
    Callable
        Wrapped function with transparent caching.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key_data = json.dumps(
                {"fn": func.__qualname__, "args": args, "kwargs": kwargs},
                sort_keys=True,
                default=str,
            )
            key = hashlib.sha256(key_data.encode()).hexdigest()
            cache = get_cache()
            result = cache.get(key)
            if result is not None:
                logger.debug("Cache hit for %s", func.__qualname__)
                return result
            result = func(*args, **kwargs)
            cache.set(key, result, expire=ttl)
            logger.debug("Cache set for %s (ttl=%ds)", func.__qualname__, ttl)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
