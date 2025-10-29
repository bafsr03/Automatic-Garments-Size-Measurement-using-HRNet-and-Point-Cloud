import time
from typing import Optional

import jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .config import settings


auth_scheme = HTTPBearer(auto_error=True)


def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)) -> dict:
    token = credentials.credentials
    try:
        options = {"require": []}
        decoded = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALG],
            audience=settings.JWT_AUD,
            issuer=settings.JWT_ISS,
            options=options,
        )
        # Exp validation happens in jwt lib if present
        return decoded
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def create_jwt(sub: str, ttl_seconds: int = 3600, aud: Optional[str] = None, iss: Optional[str] = None) -> str:
    now = int(time.time())
    payload = {"sub": sub, "iat": now, "exp": now + ttl_seconds}
    if aud or settings.JWT_AUD:
        payload["aud"] = aud or settings.JWT_AUD
    if iss or settings.JWT_ISS:
        payload["iss"] = iss or settings.JWT_ISS
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALG)


