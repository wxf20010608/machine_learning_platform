#!/bin/bash
set -e

# 不停机更新脚本 - 实现蓝绿部署

# 显示帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo "选项:"
    echo "  -h, --help     显示帮助信息"
    echo "  -p, --pull     拉取最新代码"
    echo "  --no-build     跳过构建步骤"
    echo ""
    echo "示例:"
    echo "  $0 -p          拉取最新代码并执行更新"
    echo "  $0 --no-build  执行更新但跳过构建步骤"
}

# 默认参数
PULL_CODE=false
SKIP_BUILD=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--pull)
            PULL_CODE=true
            shift
            ;;
        --no-build)
            SKIP_BUILD=true
            shift
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 确定当前活动的部署颜色
CURRENT_COLOR=$(docker exec scan_app_nginx grep -A 10 'upstream backend' /etc/nginx/conf.d/default.conf | grep -v '#' | grep 'server' | head -n 1 | grep -o 'app_[a-z]\+')

if [[ "$CURRENT_COLOR" == "app_blue" ]]; then
    CURRENT_COLOR="blue"
    NEW_COLOR="green"
    CURRENT_CONTAINER="scan_app_blue"
    NEW_CONTAINER="scan_app_green"
else
    CURRENT_COLOR="green"
    NEW_COLOR="blue"
    CURRENT_CONTAINER="scan_app_green"
    NEW_CONTAINER="scan_app_blue"
fi

echo "当前活动部署: $CURRENT_COLOR"
echo "准备部署新版本到: $NEW_COLOR"

# 拉取最新代码（如果需要）
if [[ "$PULL_CODE" == "true" ]]; then
    echo "拉取最新代码..."
    git pull
fi

# 构建新版本（如果不跳过）
if [[ "$SKIP_BUILD" == "false" ]]; then
    echo "构建新版本..."
    docker-compose build "app_$NEW_COLOR"
fi

# 启动新版本容器
echo "启动新版本容器 ($NEW_COLOR)..."
docker-compose up -d "app_$NEW_COLOR"

# 等待新容器健康检查通过
echo "等待新容器健康检查通过..."
attempts=0
max_attempts=30
while [[ $attempts -lt $max_attempts ]]; do
    if [[ $(docker inspect --format='{{.State.Health.Status}}' "$NEW_CONTAINER") == "healthy" ]]; then
        echo "新容器健康检查通过！"
        break
    fi
    echo "等待新容器健康检查通过... (${attempts}/${max_attempts})"
    sleep 5
    attempts=$((attempts+1))
done

if [[ $attempts -eq $max_attempts ]]; then
    echo "错误: 新容器健康检查未通过，回滚更新"
    docker-compose stop "app_$NEW_COLOR"
    exit 1
fi

# 更新Nginx配置，将流量切换到新容器
echo "更新Nginx配置，将流量切换到新容器..."
docker exec scan_app_nginx bash -c "cat > /etc/nginx/conf.d/default.conf << EOF
server {
    listen 80;
    server_name localhost;

    # 健康检查路径
    location /health {
        proxy_pass http://backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    # FastAPI应用
    location / {
        proxy_pass http://backend;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Streamlit应用 - KNN相机应用
    location /knn/ {
        proxy_pass http://localhost:8501/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \"upgrade\";
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }

    # Streamlit应用 - KMeans应用
    location /kmeans/ {
        proxy_pass http://localhost:8502/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \"upgrade\";
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }

    # Streamlit应用 - 线性回归应用
    location /linear/ {
        proxy_pass http://localhost:8503/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \"upgrade\";
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }

    # Streamlit应用 - 逻辑回归应用
    location /logistic/ {
        proxy_pass http://localhost:8504/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \"upgrade\";
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }

    # Streamlit应用 - 随机森林应用
    location /forest/ {
        proxy_pass http://localhost:8505/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection \"upgrade\";
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
}

# 后端服务器组 - 现在指向${NEW_COLOR}部署
upstream backend {
    server app_${NEW_COLOR}:8000;
    # ${CURRENT_COLOR}部署作为备份
    server app_${CURRENT_COLOR}:8000 backup;
}
EOF"

# 重新加载Nginx配置
echo "重新加载Nginx配置..."
docker exec scan_app_nginx nginx -s reload

echo "等待确认Nginx配置生效..."
sleep 5

echo "更新完成！新版本已成功部署到${NEW_COLOR}环境"
echo "如需回滚，请运行: $0"