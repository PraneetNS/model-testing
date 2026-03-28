from datetime import datetime, timedelta
import os
from cryptography.fernet import Fernet
from typing import Any, Union
from jose import jwt
from passlib.context import CryptContext
from app.core.config import settings

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def create_access_token(
    subject: Union[str, Any], expires_delta: timedelta = None
) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def generate_random_password() -> str:
    import secrets
    import string
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for i in range(20))

# Red Team Encryption (at-rest)
_REDTEAM_KEY = os.getenv("REDTEAM_ENCRYPTION_KEY", "7S6-vY-Z5p5A-k5p5H-k5p5H-k5p5H-k5p5H-k5p5=")
cipher = Fernet(_REDTEAM_KEY.encode())

def encrypt_content(content: str) -> bytes:
    """Encrypt sensitive red team content."""
    return cipher.encrypt(content.encode())

def decrypt_content(encrypted_content: bytes) -> str:
    """Decrypt red team content."""
    return cipher.decrypt(encrypted_content).decode()
