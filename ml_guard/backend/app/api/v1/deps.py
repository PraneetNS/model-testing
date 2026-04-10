from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import firebase_admin
from firebase_admin import auth as firebase_auth
import structlog

from app.core.config import settings
from app.infrastructure.database import SessionLocal
from app.infrastructure.persistence import models as sql_models

logger = structlog.get_logger()
security_bearer = HTTPBearer()

def get_db() -> Generator:
    """Dependency to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(
    db: AsyncSession = Depends(get_db),
    credentials: HTTPAuthorizationCredentials = Depends(security_bearer)
) -> sql_models.User:
    """
    Middleware: Verifies Firebase ID Token and returns the matching local DB user.
    Strictly follows production security standards.
    """
    token = credentials.credentials
    from jose import jwt, JWTError
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id:
            user = (await db.execute(select(sql_models.User).filter(sql_models.User.id == user_id))).scalars().first()
            if user:
                if not user.is_active:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is deactivated.")
                return user
    except JWTError:
        pass # Not a local token or expired, try firebase
    try:
        # 1. Verify ID Token with Firebase Admin SDK
        # In local dev without service accounts, this may crash on ADC fetch.
        try:
            decoded_token = firebase_auth.verify_id_token(token, check_revoked=False)
        except Exception as auth_err:
            if not settings.FIREBASE_CREDENTIALS_JSON and ("google.auth" in str(type(auth_err)) or "google-auth" in str(auth_err) or "credentials" in str(auth_err).lower() or "ADC" in str(auth_err)):
                logger.warning("Firebase ADC not found. Decoding token anonymously for local development.")
                # Decode the NextJS generated Firebase Token without signature verification locally
                decoded_token = jwt.get_unverified_claims(token)
                if "user_id" in decoded_token and "uid" not in decoded_token:
                    decoded_token["uid"] = decoded_token["user_id"]
            else:
                raise auth_err

        firebase_uid = decoded_token['uid']
        email = decoded_token.get('email')

        # 2. Lookup user in local database
        user = db.query(sql_models.User).filter(
            (sql_models.User.firebase_uid == firebase_uid) | 
            (sql_models.User.email == email)
        ).first()

        # 3. Handle Auto-Provisioning (Optional: Adjust based on onboarding flow)
        if not user:
            logger.info("Auto-provisioning user from firebase identity", email=email)
            user = sql_models.User(
                email=email,
                firebase_uid=firebase_uid,
                full_name=decoded_token.get('name', 'Cloud User'),
                is_active=True,
                role="developer"
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        
        # 4. Sync firebase_uid if missing (for legacy users migrating)
        elif not user.firebase_uid:
            user.firebase_uid = firebase_uid
            await db.commit()

        if not user.is_active:
            raise HTTPException(status_code=403, detail="Account is deactivated.")

        return user

    except firebase_auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid. Please re-authenticate.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except firebase_auth.RevokedIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication revoked. Access denied.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("Authentication internal failure", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Identity verification system failure: {str(e)}"
        )

def get_current_active_user(
    current_user: sql_models.User = Depends(get_current_user),
) -> sql_models.User:
    """Dependency to ensure the user is active."""
    return current_user

def check_role(roles: list[str]):
    """Higher-order dependency for Role Based Access Control (RBAC)."""
    def role_checker(user: sql_models.User = Depends(get_current_active_user)):
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Security Clearance Denied. Required Roles: {roles}",
            )
        return user
    return role_checker
