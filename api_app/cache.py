import hashlib
import json
import time
from typing import Any, Dict, Optional, Tuple

from .config import settings


_cache: Dict[str, Tuple[float, Any]] = {}


def _key_for(payload: Dict[str, Any]) -> str:
    # stable json hash key
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def get(payload: Dict[str, Any]) -> Optional[Any]:
    key = _key_for(payload)
    entry = _cache.get(key)
    if not entry:
        return None
    expires_at, value = entry
    if time.time() > expires_at:
        _cache.pop(key, None)
        return None
    return value


def set(payload: Dict[str, Any], value: Any) -> None:
    # enforce max items
    if len(_cache) >= settings.CACHE_MAX_ITEMS:
        # drop oldest
        oldest_key = min(_cache.items(), key=lambda kv: kv[1][0])[0]
        _cache.pop(oldest_key, None)
    key = _key_for(payload)
    _cache[key] = (time.time() + settings.CACHE_TTL_SECONDS, value)


