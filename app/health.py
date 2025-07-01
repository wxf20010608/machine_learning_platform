from fastapi import APIRouter
import os

router = APIRouter()

@router.get("/health")
async def health_check():
    """
    健康检查端点，用于Docker容器的健康监控
    返回应用的当前状态和部署颜色
    """
    return {
        "status": "healthy", 
        "color": os.getenv("DEPLOYMENT_COLOR", "unknown"),
        "version": os.getenv("APP_VERSION", "1.0.0")
    }