import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from app.db import Base

load_dotenv()  # 加载.env文件
SECRET_KEY = "your_secret_key"
ALGORITHM = "your_algorithm"

class Settings(BaseSettings):
    # 应用设置
    APP_NAME: str = "扫描登录系统"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")

    # 数据库设置
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./app.db")

    GITHUB_CLIENT_ID: str = os.getenv("GITHUB_CLIENT_ID", "")
    GITHUB_CLIENT_SECRET: str = os.getenv("GITHUB_CLIENT_SECRET", "")
    GITHUB_REDIRECT_URI: str = os.getenv("GITHUB_REDIRECT_URI", "http://127.0.0.1:8000/auth/github/callback")

    # 前端URL
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://127.0.0.1:3000")
    SCAN_PAGE_URL: str = os.getenv("SCAN_PAGE_URL", "/scan")  # 扫描界面路径

    # 邮件设置
    SMTP_SERVER: str = os.getenv("SMTP_SERVER", "smtp.qq.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "465"))
    SMTP_USER: str = os.getenv("SMTP_USER", "2140717632@qq.com")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "emgwncdmfqwyeifc")

    # JWT设置
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", SECRET_KEY)
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # 上传文件设置
    UPLOAD_DIR: str = "static/uploads"
    # 模型保存路径
    MODEL_SAVE_PATH: str = os.getenv("MODEL_SAVE_PATH", "models")

    # 跨域设置
    CORS_ORIGINS: list = ["*"]

settings = Settings()
