#!/bin/bash
set -e

# 启动FastAPI应用
echo "Starting FastAPI application..."
cd /app1
uvicorn main:app --host 0.0.0.0 --port 8000 &

# 等待FastAPI应用启动
sleep 5

# 启动Streamlit应用
echo "Starting Streamlit applications..."
python -m app.streamlit_apps.launcher &

# 保持容器运行
echo "All services started successfully!"
tail -f /dev/null