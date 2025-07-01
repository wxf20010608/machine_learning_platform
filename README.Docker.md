# Docker部署与不停机更新指南

本文档提供了使用Docker部署扫描应用系统以及实现不停机更新的详细说明。

## 目录结构

```
.
├── Dockerfile          # 定义应用容器镜像
├── docker-compose.yml  # 定义多容器应用
├── docker-entrypoint.sh # 容器启动脚本
├── update.sh           # 不停机更新脚本
└── nginx/              # Nginx配置目录
    ├── nginx.conf      # Nginx主配置
    └── conf.d/         # Nginx站点配置
        └── default.conf # 应用反向代理配置
```

## 部署架构

本项目采用蓝绿部署策略实现不停机更新：

1. **蓝绿部署**：维护两个相同的生产环境（蓝色和绿色），一个处于活动状态，另一个处于待机状态
2. **Nginx反向代理**：负责流量分发，可以无缝切换流量从蓝色环境到绿色环境（或反之）
3. **共享存储**：确保两个环境可以访问相同的数据和模型文件

## 初始部署

### 前置条件

- 安装Docker和Docker Compose
- 确保80端口和8501-8505端口未被占用

### 部署步骤

1. 创建必要的目录：

```bash
mkdir -p nginx/conf.d
```

2. 构建并启动应用：

```bash
docker-compose up -d
```

初始部署将启动蓝色环境和Nginx代理。应用将在以下地址可用：

- 主应用：http://localhost/
- KNN应用：http://localhost/knn/
- KMeans应用：http://localhost/kmeans/
- 线性回归应用：http://localhost/linear/
- 逻辑回归应用：http://localhost/logistic/
- 随机森林应用：http://localhost/forest/

## 不停机更新

当需要更新应用时，可以使用提供的`update.sh`脚本实现不停机更新：

```bash
# 赋予脚本执行权限
chmod +x update.sh

# 执行更新（拉取最新代码并部署）
./update.sh -p

# 仅执行部署（不拉取代码）
./update.sh

# 跳过构建步骤
./update.sh --no-build
```

更新过程：

1. 脚本自动检测当前活动环境（蓝色或绿色）
2. 构建并启动非活动环境的新版本
3. 等待新版本通过健康检查
4. 更新Nginx配置，将流量切换到新版本
5. 保留旧版本作为备份，以便快速回滚

## 回滚操作

如果新版本出现问题，可以通过再次运行更新脚本快速回滚：

```bash
./update.sh --no-build
```

这将切换回之前的活动环境，无需重新构建。

## 环境变量配置

应用的环境变量可以在`docker-compose.yml`文件中的`environment`部分进行配置。重要的环境变量包括：

- `DATABASE_URL`：数据库连接字符串
- `APP_ENV`：应用环境（development/production）
- `DEPLOYMENT_COLOR`：部署颜色（blue/green）

## 注意事项

1. 确保`docker-entrypoint.sh`脚本具有执行权限
2. 在Windows环境中，可能需要调整脚本中的行尾序列（CRLF -> LF）
3. 数据和模型文件存储在挂载卷中，确保这些目录具有适当的权限

## 故障排除

- 检查容器日志：`docker-compose logs -f`
- 检查容器健康状态：`docker ps`
- 检查Nginx配置：`docker exec scan_app_nginx nginx -t`