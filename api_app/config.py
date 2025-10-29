import os
from typing import Optional


class Settings:
    # Security
    JWT_SECRET: str = os.getenv("JWT_SECRET", "dev-secret-change-me")
    JWT_ALG: str = os.getenv("JWT_ALG", "HS256")
    JWT_AUD: Optional[str] = os.getenv("JWT_AUD")
    JWT_ISS: Optional[str] = os.getenv("JWT_ISS")
    # Rate limiting
    RATE_LIMIT_PER_MIN: int = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
    RATE_LIMIT_BURST: int = int(os.getenv("RATE_LIMIT_BURST", "30"))
    # Caching
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "600"))
    CACHE_MAX_ITEMS: int = int(os.getenv("CACHE_MAX_ITEMS", "256"))
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    # API
    API_PREFIX: str = os.getenv("API_PREFIX", "/v1")


settings = Settings()


