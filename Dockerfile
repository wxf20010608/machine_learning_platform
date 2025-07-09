FROM python:3.9-slim

# 设置工作目录
WORKDIR /app1

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app1

# 配置apt镜像源 - 使用sources.list.d目录
RUN echo "deb https://mirrors.aliyun.com/debian/ bullseye main non-free contrib" > /etc/apt/sources.list.d/aliyun.list && \
    echo "deb https://mirrors.aliyun.com/debian-security/ bullseye-security main" >> /etc/apt/sources.list.d/aliyun.list && \
    echo "deb https://mirrors.aliyun.com/debian/ bullseye-updates main non-free contrib" >> /etc/apt/sources.list.d/aliyun.list

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopencv-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt
COPY requirements.txt .

# 配置pip镜像源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com && \
    pip config set global.timeout 300 && \
    pip config set global.retries 10

# 安装所有Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . /app1/

# 创建必要的目录
RUN mkdir -p /app1/static/uploads /app1/models

# 暴露端口 (FastAPI和Streamlit应用)
EXPOSE 8000 8501 8502 8503 8504 8505

# 设置健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["sh", "/app1/docker-entrypoint.sh"]