# 机器学习平台

一个基于FastAPI、Streamlit、PyTorch、OpenCV和Docker的综合机器学习平台，提供多种机器学习算法的可视化界面和API接口。

## 项目概述

本项目是一个集成了多种机器学习算法的Web应用平台，支持通过友好的用户界面进行模型训练、预测和可视化。平台采用前后端分离架构，后端使用FastAPI提供RESTful API服务，前端使用Streamlit构建交互式应用界面。

### 主要功能

- **用户认证系统**：支持邮箱验证码和GitHub OAuth登录
- **文档扫描处理**：基于OpenCV的图像处理和文档扫描功能
- **机器学习算法**：
  - KNN分类器（支持摄像头实时识别）
  - K-Means聚类
  - 逻辑回归
  - 线性回归
  - 随机森林
- **模型训练与预测**：支持通过Web界面和API接口进行模型训练和预测
- **数据可视化**：使用Streamlit提供直观的数据可视化界面

## 技术栈

- **后端框架**：FastAPI
- **前端框架**：Streamlit
- **机器学习库**：PyTorch, scikit-learn
- **图像处理**：OpenCV
- **模型序列化**：joblib
- **容器化**：Docker
- **反向代理**：Nginx
- **数据库**：SQLite

## 项目结构

```
.
├── app/                    # 主应用目录
│   ├── auth/              # 认证相关模块
│   ├── scan/              # 文档扫描处理模块
│   ├── streamlit_apps/    # Streamlit应用
│   ├── ml_models.py       # 机器学习模型定义
│   └── ml_routers.py      # 机器学习API路由
├── static/                # 静态文件
├── templates/             # HTML模板
├── models/                # 保存的模型
├── data/                  # 数据集
├── nginx/                 # Nginx配置
├── .env                   # 环境变量配置
├── main.py                # 主应用入口
├── Dockerfile             # Docker构建文件
├── docker-compose.yml     # Docker Compose配置
├── docker-entrypoint.sh   # Docker启动脚本
└── requirements.txt       # 依赖包列表
```

## 安装与部署

### 本地开发环境

1. 克隆仓库

```bash
git clone <repository-url>
cd Machine_learning_platform
```

2. 创建并激活虚拟环境

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 配置环境变量

复制`.env.example`文件为`.env`，并根据需要修改配置。

5. 启动应用

```bash
python main.py
```

应用将在 http://localhost:8000 上运行。

### Docker部署

1. 使用Docker Compose启动应用

```bash
docker-compose up -d
```

应用将在 http://localhost 上运行。

2. 不停机更新（蓝绿部署）

```bash
# 赋予脚本执行权限
chmod +x update.sh

# 执行更新
./update.sh
```

详细的Docker部署说明请参考 [README.Docker.md](README.Docker.md)。

## 使用指南

### 主要功能模块

1. **用户认证**
   - 访问首页 http://localhost:8000 进行登录
   - 支持邮箱验证码和GitHub OAuth登录

2. **文档扫描**
   - 访问 http://localhost:8000/scan 上传并处理文档图像

3. **机器学习应用**
   - 登录后访问 http://localhost:8000/features 查看所有可用的机器学习功能
   - 点击相应卡片进入各个机器学习应用

### 机器学习应用说明

1. **KNN摄像头识别**
   - 实时捕获摄像头画面并进行分类
   - 支持添加自定义类别样本
   - 使用MobileNetV2提取特征

2. **K-Means聚类**
   - 支持上传数据进行聚类分析
   - 可视化聚类结果

3. **逻辑回归**
   - 基于PyTorch实现的逻辑回归模型
   - 支持二分类任务

4. **线性回归**
   - 基于PyTorch实现的线性回归模型
   - 支持加州房价数据集预测

5. **随机森林**
   - 基于神经网络实现的随机森林近似模型
   - 支持多分类任务

## API接口

平台提供了一系列RESTful API接口，用于模型训练和预测：

- `/ml/knn/train` - 训练KNN模型
- `/ml/knn/predict` - 使用KNN模型进行预测
- `/ml/kmeans/train` - 训练K-Means模型
- `/ml/kmeans/predict` - 使用K-Means模型进行聚类
- `/ml/logistic_regression/train` - 训练逻辑回归模型
- `/ml/logistic_regression/predict` - 使用逻辑回归模型进行预测
- `/ml/linear_regression/train` - 训练线性回归模型
- `/ml/linear_regression/predict` - 使用线性回归模型进行预测
- `/ml/random_forest/train` - 训练随机森林模型
- `/ml/random_forest/predict` - 使用随机森林模型进行预测

所有API接口都需要JWT令牌认证。

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建Pull Request

## 许可证

本项目采用 MIT 许可证 - 详情请参阅 [LICENSE](LICENSE) 文件。

## 联系方式

如有任何问题或建议，请通过以下方式联系我们：

- 邮箱：2140717632@qq.com
- GitHub Issues: [项目Issues页面](https://github.com/wangxianfu/Machine_learning_platform/issues)