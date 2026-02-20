from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.api.v1 import deps
from app.core import security
from app.core.config import settings
from app.infrastructure.persistence import models as sql_models
from app.domain.models import user as user_schema

router = APIRouter()


@router.post("/login", response_model=user_schema.Token)
def login_access_token(
    db: Session = Depends(deps.get_db), form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user = db.query(sql_models.User).filter(sql_models.User.email == form_data.username).first()
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
def register_user(
    *,
    db: Session = Depends(deps.get_db),
    user_in: user_schema.UserCreate
) -> Any:
    """
    Create new user and tenant.
    """
    user = db.query(sql_models.User).filter(sql_models.User.email == user_in.email).first()
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this username already exists in the system.",
        )
    
    # Create Tenant
    tenant = db.query(sql_models.Tenant).filter(sql_models.Tenant.name == user_in.tenant_name).first()
    if not tenant:
        tenant = sql_models.Tenant(name=user_in.tenant_name)
        db.add(tenant)
        db.commit()
        db.refresh(tenant)

    # Create User
    db_user = sql_models.User(
        email=user_in.email,
        hashed_password=security.get_password_hash(user_in.password),
        full_name=user_in.full_name,
        role=user_in.role,
        tenant_id=tenant.id
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
