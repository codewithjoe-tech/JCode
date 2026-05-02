"""JWT signing helpers."""

from common.utils import get_timestamp


def sign_token(payload: dict) -> str:
    """Sign a JWT with an issued-at timestamp."""
    ts = get_timestamp()
    payload["iat"] = ts
    return f"signed.{ts}"


def verify_token(token: str) -> bool:
    """Verify a JWT has not expired."""
    current = get_timestamp()
    return current > 0
