import random
import string
from datetime import datetime, timedelta

import requests
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.db import User, get_db
from app.config import settings
from app.auth.models import TokenData

# 密码上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 密码Bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


def verify_password(plain_password, hashed_password):
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """生成密码哈希"""
    return pwd_context.hash(password)


def generate_verification_code(length=6):
    """生成随机验证码"""
    return ''.join(random.choices(string.digits, k=length))


def store_verification_code(db: Session, email: str, code: str):
    """存储验证码"""
    user = db.query(User).filter(User.email == email).first()

    if not user:
        # 如果用户不存在，创建一个新用户
        user = User(
            email=email,
            verification_code=code,
            verification_code_expires=datetime.utcnow() + timedelta(minutes=10)
        )
        db.add(user)
    else:
        # 更新现有用户的验证码
        user.verification_code = code
        user.verification_code_expires = datetime.utcnow() + timedelta(minutes=10)

    db.commit()
    return user


def verify_code(db: Session, email: str, code: str):
    """验证验证码"""
    user = db.query(User).filter(User.email == email).first()

    if not user:
        return False

    if user.verification_code != code:
        return False

    if datetime.utcnow() > user.verification_code_expires:
        return False

    # 验证成功后清除验证码
    user.verification_code = None
    user.verification_code_expires = None
    db.commit()

    return True


def create_access_token(data: dict, expires_delta: timedelta = None):
    """创建访问令牌"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)

    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """获取当前用户"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭据",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == token_data.email).first()
    if user is None:
        raise credentials_exception

    return user


async def github_login(code: str, db: Session):
    """使用GitHub授权码获取访问令牌和用户信息"""
    # 获取GitHub访问令牌
    token_url = "https://github.com/login/oauth/access_token"
    headers = {"Accept": "application/json"}
    data = {
        "client_id": settings.GITHUB_CLIENT_ID,
        "client_secret": settings.GITHUB_CLIENT_SECRET,
        "code": code,
        "redirect_uri": settings.GITHUB_REDIRECT_URI
    }

    response = requests.post(token_url, headers=headers, data=data)
    response_json = response.json()

    if "error" in response_json:
        return None

    access_token = response_json.get("access_token")

    # 获取GitHub用户信息
    user_url = "https://api.github.com/user"
    headers = {
        "Authorization": f"token {access_token}",
        "Accept": "application/json"
    }

    user_response = requests.get(user_url, headers=headers)
    user_info = user_response.json()

    # 获取用户邮箱
    email = user_info.get("email")

    # 如果GitHub用户没有公开邮箱，则获取主要邮箱
    if not email:
        emails_url = "https://api.github.com/user/emails"
        emails_response = requests.get(emails_url, headers=headers)
        emails_info = emails_response.json()

        # 查找主要邮箱
        for email_obj in emails_info:
            if email_obj.get("primary"):
                email = email_obj.get("email")
                break

    if not email:
        return None

    # 检查用户是否存在，不存在则创建
    user = db.query(User).filter(User.email == email).first()
    if not user:
        user = User(
            email=email,
            is_active=True,
            # 可以保存更多GitHub信息，如用户名等
            hashed_password=None  # GitHub用户不需要密码
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    # 创建访问令牌
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": email},
        expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }