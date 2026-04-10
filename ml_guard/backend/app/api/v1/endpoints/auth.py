from datetime import timedelta
from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.api.v1 import deps
from app.core import security
from app.core.config import settings
from app.infrastructure.persistence import models as sql_models
from app.domain.models import user as user_schema

from pydantic import BaseModel

class FirebaseLoginRequest(BaseModel):
    id_token: str

router = APIRouter()

@router.post("/firebase/login")
async def firebase_login(
    data: FirebaseLoginRequest,
    db: AsyncSession = Depends(deps.get_db)
):
    """
    Authenticate a user via Firebase ID Token.
    Returns a local JWT for session management.
    """
    try:
        from firebase_admin import auth as firebase_auth
        # Verify the ID token sent by the client
        decoded_token = firebase_auth.verify_id_token(data.id_token)
        uid = decoded_token['uid']
        email = decoded_token.get('email')
        name = decoded_token.get('name', 'Firebase User')
        
        if not email:
            raise HTTPException(status_code=400, detail="Firebase token missing email")

        # Check if user exists, else create (Fireflink Philosophy: Just-In-Time Provisioning)
        user = (await db.execute(select(sql_models.User).filter(sql_models.User.email == email))).scalars().first()
        if not user:
            # Create a default tenant for new users if needed
            tenant = (await db.execute(select(sql_models.Tenant))).scalars().first()
            if not tenant:
                tenant = sql_models.Tenant(name="Default Organization")
                db.add(tenant)
                db.flush()
            
            user = sql_models.User(
                email=email,
                full_name=name,
                hashed_password=security.get_password_hash(security.generate_random_password()),
                tenant_id=tenant.id,
                is_active=True
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        return {
            "access_token": security.create_access_token(
                subject=str(user.id), expires_delta=access_token_expires
            ),
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Firebase token: {str(e)}"
        )


@router.post("/login", response_model=user_schema.Token)
async def login_access_token(
    db: AsyncSession = Depends(deps.get_db), form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user = (await db.execute(select(sql_models.User).filter(sql_models.User.email == form_data.username))).scalars().first()
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    elif not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return {
        "access_token": security.create_access_token(
            user.id, expires_delta=access_token_expires
        ),
        "token_type": "bearer",
    }


@router.post("/register", response_model=user_schema.UserInDB)
async def register_user(
    *,
    db: AsyncSession = Depends(deps.get_db),
    user_in: user_schema.UserCreate
) -> Any:
    """
    Create new user and tenant.
    """
    user = (await db.execute(select(sql_models.User).filter(sql_models.User.email == user_in.email))).scalars().first()
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this username already exists in the system.",
        )
    
    # Create Tenant
    tenant = (await db.execute(select(sql_models.Tenant).filter(sql_models.Tenant.name == user_in.tenant_name))).scalars().first()
    if not tenant:
        tenant = sql_models.Tenant(name=user_in.tenant_name)
        db.add(tenant)
        await db.commit()
        await db.refresh(tenant)

    # Create User
    db_user = sql_models.User(
        email=user_in.email,
        hashed_password=security.get_password_hash(user_in.password),
        full_name=user_in.full_name,
        role=user_in.role,
        tenant_id=tenant.id
    )
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    return db_user
