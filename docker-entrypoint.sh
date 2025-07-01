#!/bin/bash
set -e

# 创建健康检查端点
echo "Creating health check endpoint..."
cat > /app/health_endpoint.py << 'EOF'
from fastapi import FastAPI, APIRouter
import uvicorn
import os

app = FastAPI()
router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy", "color": os.getenv("DEPLOYMENT_COLOR", "unknown")}

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("health_endpoint:app", host="0.0.0.0", port=8000)
EOF

# 启动健康检查服务
python /app/health_endpoint.py &

# 等待健康检查服务启动
sleep 5

# 启动FastAPI应用
echo "Starting FastAPI application..."
uvicorn main:app --host 0.0.0.0 --port 8000 &

# 启动Streamlit应用
echo "Starting Streamlit applications..."
python -m app.streamlit_apps.launcher &

# 保持容器运行
echo "All services started successfully!"
tail -f /dev/null