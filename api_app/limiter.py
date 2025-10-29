import time
from typing import Dict, Tuple

from fastapi import Request, HTTPException

from .config import settings


_buckets: Dict[str, Tuple[float, float]] = {}


def rate_limit(request: Request) -> None:
    # simple token bucket per client ip
    ident = request.client.host if request.client else "anon"
    capacity = settings.RATE_LIMIT_BURST
    refill_per_min = settings.RATE_LIMIT_PER_MIN
    now = time.time()
    tokens, last = _buckets.get(ident, (capacity, now))
    # refill
    elapsed_min = max(0.0, (now - last) / 60.0)
    tokens = min(capacity, tokens + elapsed_min * refill_per_min)
    if tokens < 1.0:
        raise HTTPException(status_code=429, detail="Too Many Requests")
    tokens -= 1.0
    _buckets[ident] = (tokens, now)


