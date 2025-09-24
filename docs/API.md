# 机器学习平台 API 文档

> 基于 FastAPI。所有 JSON 响应字段以实际接口返回为准。本文档仅列出主要稳定端点与示例。

## 认证与用户（/auth）

- POST `/auth/send-verification-code`
  - 描述：发送邮箱验证码（后台任务发邮件，验证码有效期见服务端配置）
  - 请求体（application/json）：
    ```json
    { "email": "user@example.com" }
    ```
  - 响应：
    ```json
    { "message": "验证码已发送，请查收邮件" }
    ```

- POST `/auth/login`
  - 描述：使用邮箱验证码登录（校验成功后重定向到前端 `features` 页面携带 token）。
  - 请求体（application/json）：
    ```json
    { "email": "user@example.com", "verification_code": "123456" }
    ```
  - 响应（302 重定向）：Location: `{FRONTEND_URL}/features?token=...`

- GET `/auth/github/login`
  - 描述：GitHub OAuth 登录入口（302 跳转至 GitHub 授权页）。

- GET `/auth/github/callback?code=xxx`
  - 描述：GitHub 回调；成功后 302 跳转到 `{FRONTEND_URL}/features?token=...`

- GET `/auth/me`
  - 描述：获取当前登录用户信息
  - 认证：Bearer Token（来自登录流程生成的 JWT）
  - 响应：
    ```json
    { "id": 1, "email": "user@example.com", "is_active": true }
    ```

## 通用与页面路由

- GET `/`、`/login`
  - 描述：返回登录页面（HTML）。

- GET `/features?token=...`
  - 描述：返回功能入口页面（HTML）。

- GET `/ml-features?token=...`
  - 描述：返回 ML 功能页面（HTML）。

- Streamlit 应用跳转（要求 query 中携带 token）：
  - GET `/knn` → 302 到 `/knn/?token=...`
  - GET `/kmeans` → 302 到 `/kmeans/?token=...`
  - GET `/linear-regression` → 302 到 `/linear/?token=...`
  - GET `/logistic-regression` → 302 到 `/logistic/?token=...`
  - GET `/random-forest` → 302 到 `/forest/?token=...`

- GET `/start-streamlit-apps`
  - 描述：后台线程启动所有 Streamlit 子应用。
  - 响应：
    ```json
    { "message": "Streamlit应用已启动" }
    ```

## 健康检查（/health）

- GET `/health`
  - 描述：容器/服务健康状态探针
  - 响应：
    ```json
    { "status": "healthy", "color": "blue", "version": "1.0.0" }
    ```

## 文档扫描与图像处理（/api | /api/scan）

- POST `/api/scan`
  - 描述：上传图片并进行文档扫描/处理。
  - 请求：multipart/form-data，字段名 `file`
  - 响应（成功示例）：
    ```json
    {
      "original_image": "/static/uploads/original_20250101010101.jpg",
      "processed_image": "/static/processed/processed_xxx.jpg",
      "message": "Document processed successfully"
    }
    ```
  - 失败返回：HTTP 4xx/5xx，含 `detail` 或 `error` 字段。

- GET `/api/image/{filename:path}`
  - 描述：按路径返回静态图像文件。

- 附：页面端上传表单
  - GET `/scan` 返回上传页（HTML）
  - GET `/scan-form` 返回另一上传页面（HTML）

## 机器学习 API（/ml）

说明：所有 `/ml/**` 端点均要求 Bearer Token（来自登录流程的 JWT）。部分训练/预测依赖预先保存的模型与 scaler，路径由配置 `MODEL_SAVE_PATH` 决定。

- 训练
  - POST `/ml/knn/train`
    - query/body 参数：`n_neighbors`（默认 5）
    - 响应：`{"message": "KNN model trained successfully"}`
  - POST `/ml/kmeans/train`
    - 参数：`n_clusters`（默认 3）
    - 响应：`{"message": "KMeans model trained successfully"}`
  - POST `/ml/logistic_regression/train`
    - 参数：`epochs`（默认 1000）、`lr`（默认 0.01）
    - 响应：`{"message": "Logistic Regression model trained successfully"}`
  - POST `/ml/linear_regression/train`
    - 参数：`epochs`（默认 1000）、`lr`（默认 0.01）
    - 响应：`{"message": "Linear Regression model trained successfully"}`
  - POST `/ml/random_forest/train`
    - 参数：`epochs`（默认 100）、`lr`（默认 0.01）
    - 响应：`{"message": "Random Forest model trained successfully"}`

- 预测（均为 `multipart/form-data`，字段名 `file`，内容为 CSV）
  - POST `/ml/knn/predict`
    - 响应：`{"predictions": [0,1,1,...]}`
  - POST `/ml/kmeans/predict`
    - 响应：`{"clusters": [0,2,1,...]}`
  - POST `/ml/logistic_regression/predict`
    - 响应：`{"predictions": [0,1,0,...]}`
  - POST `/ml/linear_regression/predict`
    - 响应：`{"predictions": [3.14, 2.71, ...]}`
  - POST `/ml/random_forest/predict`
    - 响应：
      ```json
      {
        "predictions": [0, 2, 1, ...],
        "class_names": ["setosa", "versicolor", "virginica"]
      }
      ```

## 静态资源

- GET `/static/...`
  - 描述：静态文件（上传文件、处理结果、前端资源等）。

## 认证说明（JWT）

- 获取方式：通过 `/auth/login`（邮箱验证码）或 GitHub OAuth 登录成功后由后端签发。
- 用法：在需要鉴权的接口（如 `/ml/**`）请求头携带：
  ```http
  Authorization: Bearer <JWT_TOKEN>
  ```

## 错误响应格式

- FastAPI 默认错误：
  ```json
  { "detail": "错误描述" }
  ```
- 项目中部分接口也可能返回：
  ```json
  { "error": "错误描述" }
  ```

## 示例：curl 请求

- 发送验证码：
  ```bash
  curl -X POST http://localhost:8000/auth/send-verification-code \
       -H "Content-Type: application/json" \
       -d '{"email":"user@example.com"}'
  ```

- 扫描文件：
  ```bash
  curl -X POST http://localhost:8000/api/scan \
       -F file=@/path/to/image.jpg
  ```

- 训练 KNN：
  ```bash
  curl -X POST "http://localhost:8000/ml/knn/train?n_neighbors=5" \
       -H "Authorization: Bearer $TOKEN"
  ```

- 预测 KMeans：
  ```bash
  curl -X POST http://localhost:8000/ml/kmeans/predict \
       -H "Authorization: Bearer $TOKEN" \
       -F file=@/path/to/data.csv
  ```
