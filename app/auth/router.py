from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse

from app.auth.github import get_github_access_token, get_github_user
from app.db import get_db
from app.email_utils import send_verification_email
from app.auth.models import EmailVerificationRequest, VerificationLoginRequest, Token, GitHubLoginRequest
from app.auth.service import (
    generate_verification_code,
    store_verification_code,
    verify_code,
    create_access_token, github_login, get_current_user
)
from datetime import timedelta
from app.config import settings

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/send-verification-code")
async def send_code(
        request: EmailVerificationRequest,
        background_tasks: BackgroundTasks,
        db: Session = Depends(get_db)
):
    """发送邮箱验证码"""
    # 生成6位数验证码
    verification_code = generate_verification_code()

    # 存储验证码
    store_verification_code(db, request.email, verification_code)

    # 后台任务发送邮件
    background_tasks.add_task(
        send_verification_email,
        email_to=request.email,
        verification_code=verification_code
    )

    return {"message": "验证码已发送，请查收邮件"}


# @router.post("/login", response_model=Token)
# async def login_with_verification_code(
#         request: VerificationLoginRequest,
#         db: Session = Depends(get_db)
# ):
#     """使用邮箱验证码登录"""
#     # 验证验证码
#     is_valid = verify_code(db, request.email, request.verification_code)
#
#     if not is_valid:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="验证码无效或已过期",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#
#     # 创建访问令牌
#     access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": request.email},
#         expires_delta=access_token_expires
#     )
#
#     return {"access_token": access_token, "token_type": "bearer"}

@router.get("/github/login")
async def github_oauth_init():
    return RedirectResponse(
        f"https://github.com/login/oauth/authorize?"
        f"client_id={settings.GITHUB_CLIENT_ID}&"
        f"redirect_uri={settings.GITHUB_REDIRECT_URI}&"
        f"scope=user:email&"
        f"allow_signup=true&"
        f"prompt=login&"
        f"login="  # 空值强制显示登录页面
    )


@router.get("/github/callback")
async def github_oauth_callback(code: str, db: Session = Depends(get_db)):
    """GitHub 授权回调"""
    try:
        result = await github_login(code, db)
        if not result:
            # GitHub登录失败，重定向到登录页面并显示错误信息
            return RedirectResponse(
                url=f"{settings.FRONTEND_URL}/login?error=github_login_failed"
            )

        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/features?token={result['access_token']}"
        )
    except Exception as e:
        # 捕获所有异常，确保用户体验不中断
        print(f"GitHub OAuth回调错误: {e}")
        return RedirectResponse(
            url=f"{settings.FRONTEND_URL}/login?error=github_login_error&message={str(e)}"
        )


@router.post("/login", response_model=Token)
async def login_with_verification_code(
        request: VerificationLoginRequest,
        db: Session = Depends(get_db)
):
    """使用邮箱验证码登录"""
    # 验证验证码
    is_valid = verify_code(db, request.email, request.verification_code)

    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="验证码无效或已过期",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 创建访问令牌
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": request.email},
        expires_delta=access_token_expires
    )

    return RedirectResponse(
        url=f"{settings.FRONTEND_URL}/features?token={access_token}"
    )


# @router.get("/github/callback")
# async def github_oauth_callback(code: str, db: Session = Depends(get_db)):
#     """GitHub 授权回调"""
#     result = await github_login(code, db)
#     if not result:
#         # 登录失败，重定向到登录页并显示错误
#         return RedirectResponse(
#             url=f"{settings.FRONTEND_URL}/login?error=github_login_failed"
#         )
#
#     # 登录成功，重定向到扫描页
#     return RedirectResponse(
#         url=f"{settings.FRONTEND_URL}/scan?token={result['access_token']}"
#     )

@router.post("/github", response_model=Token)
async def github_oauth_api(request: GitHubLoginRequest, db: Session = Depends(get_db)):
    """前端直接提交 code 的登录 API"""
    return await github_login(request.code, db)

# 获取当前用户信息
@router.get("/me", response_model=dict)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """获取当前登录用户信息"""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "is_active": current_user.is_active
    }