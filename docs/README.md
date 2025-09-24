# API 文档使用指南

本项目提供了多种格式的API文档，方便不同场景下的使用。

## 📚 文档类型

### 1. 交互式在线文档 (推荐)

启动服务器后，可以通过浏览器访问以下地址：

- **Swagger UI**: http://localhost:8000/docs
  - 功能最全面的交互式API文档
  - 支持直接在浏览器中测试API
  - 包含完整的请求/响应示例

- **ReDoc**: http://localhost:8000/redoc  
  - 更美观的文档展示界面
  - 适合阅读和分享
  - 支持搜索和导航

### 2. 静态文档

- **HTML版本**: `docs/API.html`
  - 可离线查看的HTML文档
  - 包含完整的API说明和示例
  - 适合存档和分发

- **Markdown版本**: `docs/API.md`
  - 源代码格式的文档
  - 便于版本控制和协作编辑
  - 支持GitHub等平台直接显示

- **OpenAPI规范**: `docs/openapi.json`
  - 标准的OpenAPI 3.0格式
  - 可导入到其他API工具
  - 支持代码生成

## 🚀 快速开始

### 启动服务器并查看在线文档

```bash
# 1. 启动FastAPI服务器
python main.py

# 2. 访问Swagger UI
open http://localhost:8000/docs

# 3. 访问ReDoc
open http://localhost:8000/redoc
```

### 生成静态文档

```bash
# 生成HTML和OpenAPI文档
python generate_docs.py
```

## 🔧 自定义配置

### 修改API文档信息

编辑 `main.py` 中的FastAPI应用配置：

```python
app = FastAPI(
    title="您的API标题",
    description="您的API描述",
    version="您的版本号",
    docs_url="/docs",      # Swagger UI路径
    redoc_url="/redoc",    # ReDoc路径
    openapi_url="/openapi.json"  # OpenAPI JSON路径
)
```

### 添加API端点文档

在路由函数中添加详细的文档字符串：

```python
@router.post("/your-endpoint")
async def your_function(param: str):
    """
    您的API端点描述
    
    详细说明这个端点做什么，如何使用。
    
    - **param**: 参数说明
    
    返回结果的说明。
    """
    return {"result": "success"}
```

## 📋 API分类

当前API按功能分为以下几类：

- **认证 (auth)**: 用户登录、注册、OAuth
- **扫描 (scan)**: 文档处理和图像分析  
- **机器学习 (ml)**: 各种ML算法的训练和预测
- **健康检查 (health)**: 服务状态监控

## 🔐 认证说明

大部分API需要JWT Token认证：

1. 通过 `/auth/send-verification-code` 发送验证码
2. 通过 `/auth/login` 登录获取token
3. 在请求头中携带: `Authorization: Bearer <token>`

或使用GitHub OAuth:
1. 访问 `/auth/github/login` 进行GitHub授权
2. 授权成功自动获取token

## 🛠️ 开发工具集成

### Postman

1. 访问 http://localhost:8000/openapi.json
2. 复制OpenAPI JSON内容
3. 在Postman中导入OpenAPI规范

### Insomnia

1. 访问 http://localhost:8000/openapi.json
2. 使用Insomnia的"Import from URL"功能
3. 输入OpenAPI JSON的URL

### VS Code REST Client

在 `.http` 文件中使用：

```http
### 健康检查
GET http://localhost:8000/health

### 发送验证码
POST http://localhost:8000/auth/send-verification-code
Content-Type: application/json

{
  "email": "test@example.com"
}

### 登录 (替换为实际验证码)
POST http://localhost:8000/auth/login
Content-Type: application/json

{
  "email": "test@example.com",
  "verification_code": "123456"
}
```

## 📝 文档维护

### 更新API文档

1. 修改路由函数的文档字符串
2. 重启服务器使更改生效
3. 重新生成静态文档（如需要）

### 添加新的API端点

1. 在相应的路由文件中添加端点
2. 编写详细的文档字符串
3. 更新 `docs/API.md` 中的端点列表
4. 重新生成静态文档

## 🐛 故障排除

### 常见问题

**Q: 无法访问 /docs 页面**
A: 确保服务器正在运行，检查端口8000是否被占用

**Q: OpenAPI文档显示不完整**
A: 检查路由函数的文档字符串格式，确保使用正确的markdown语法

**Q: 静态文档生成失败**
A: 安装所需依赖：`pip install requests markdown jinja2`

### 获取帮助

如果遇到问题，请：

1. 检查服务器日志输出
2. 确认所有依赖已正确安装
3. 查看FastAPI官方文档: https://fastapi.tiangolo.com/

## 📄 许可证

本项目的API文档遵循与主项目相同的许可证。
