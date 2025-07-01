import requests
from fastapi import HTTPException

from app.config import settings

# 7d01fb501176526d498cd385082145e2b430d5b9
async def get_github_access_token(code: str) -> str:
    """通过GitHub授权码获取访问令牌"""

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
        raise HTTPException(
            status_code=400,
            detail=f"GitHub OAuth 错误: {response_json.get('error_description', '未知错误')}"
        )

    return response_json.get("access_token")


async def get_github_user(access_token: str) -> dict:
    """通过访问令牌获取GitHub用户信息"""

    user_url = "https://api.github.com/user"
    headers = {
        "Authorization": f"token {access_token}",
        "Accept": "application/json"
    }

    response = requests.get(user_url, headers=headers)
    user_data = response.json()

    if "message" in user_data and user_data["message"] == "Bad credentials":
        raise HTTPException(
            status_code=401,
            detail="GitHub认证失败: 无效的访问令牌"
        )

    # 获取用户邮箱
    emails_url = "https://api.github.com/user/emails"
    emails_response = requests.get(emails_url, headers=headers)
    emails_data = emails_response.json()

    # 提取主要邮箱
    primary_email = None
    for email in emails_data:
        if email.get("primary"):
            primary_email = email.get("email")
            break

    # 合并信息
    user_data["email"] = primary_email or user_data.get("email")

    return user_data
