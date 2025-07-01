from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from app.db import Base, engine
from app.email_utils import send_verification_email
from app.scan.processor import process_document_image
from app.auth.router import router as auth_router
from app.health import router as health_router
from app.scan.router import router as scan_router
from app.ml_routers import router as ml_router
import logging
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载.env文件中的配置
load_dotenv()

# 配置常量
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default_fallback_key_1234567890")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 确保静态文件目录存在
STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
PROCESSED_DIR = STATIC_DIR / "processed"
MODELS_DIR = STATIC_DIR / "models"

for directory in [STATIC_DIR, UPLOADS_DIR, PROCESSED_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时创建表
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(lifespan=lifespan)

# 配置静态文件服务
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 配置模板目录
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置为具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含所有路由
app.include_router(auth_router)
app.include_router(health_router)
app.include_router(scan_router)
app.include_router(ml_router)

# OAuth2 scheme 用于验证令牌
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

router = APIRouter()

# 设置密钥，提供一个默认值以确保始终有值
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "default_fallback_key_1234567890")

# 添加机器学习功能页面路由
@app.get("/ml-features")
async def ml_features_page(request: Request):
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    return templates.TemplateResponse("features.html", {"request": request})

# 添加Streamlit应用路由
@app.get("/knn")
async def knn_app(request: Request):
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    return {"url": f"http://localhost:8501?token={token}"}  # 使用PyTorch版本的KNN应用

@app.get("/kmeans")
async def kmeans_app(request: Request):
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    return {"url": f"http://localhost:8502?token={token}"}  # 直接返回完整URL

@app.get("/linear-regression")
async def logistic_regression_app(request: Request):
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    return {"url": f"http://localhost:8503?token={token}"}  # 修正为8503端口

@app.get("/logistic-regression")
async def linear_regression_app(request: Request):
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    return {"url": f"http://localhost:8504?token={token}"}  # 修正为8504端口

@app.get("/random-forest")
async def random_forest_app(request: Request):
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    return {"url": f"http://localhost:8505?token={token}"}  # 直接返回完整URL

# 添加启动Streamlit应用的路由
@app.get("/start-streamlit-apps")
async def start_streamlit_apps():
    try:
        from app.streamlit_apps.launcher import main as start_apps
        import threading
        # 在后台线程中启动Streamlit应用
        threading.Thread(target=start_apps, daemon=True).start()
        return {"message": "Streamlit应用已启动"}
    except Exception as e:
        logger.error(f"启动Streamlit应用时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动Streamlit应用失败: {str(e)}")

# JWT 令牌创建函数
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})

    # 使用单一参数 - 适用于旧版本的python-jose
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, ALGORITHM)
    return encoded_jwt

# 验证令牌函数
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 使用单一参数 - 适用于旧版本的python-jose
        payload = jwt.decode(token, SECRET_KEY, ALGORITHM)
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        return {"email": email}
    except JWTError:
        raise credentials_exception

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/scan")
async def scan_page(request: Request):
    return templates.TemplateResponse("scan.html", {"request": request})

@app.get("/scan/app", tags=["scan"])
async def scan_app(request: Request):
    return templates.TemplateResponse("scan_app.html", {"request": request})

# 发送验证码路由
@app.post("/send-verification-code/")
async def send_verification_code(email: str = Form(...)):
    logger.info(f"收到发送验证码请求: {email}")

    # 生成验证码
    verification_code = generate_verification_code()
    logger.info(f"生成验证码: {verification_code} 发送到: {email}")

    # 存储验证码
    verification_codes[email] = verification_code

    # 发送邮件
    result = send_verification_email(email, verification_code)

    if result:
        logger.info(f"成功发送验证码到 {email}")
        return {"message": "验证码已成功发送"}
    else:
        logger.error(f"发送验证码到 {email} 失败")
        raise HTTPException(status_code=500, detail="发送验证码失败，请检查服务器日志")

# 验证码验证路由
@app.post("/verify-code/")
async def verify_code(email: str = Form(...), code: str = Form(...)):
    # 验证用户提供的验证码
    if verify_user_code(email, code):  # 您需要实现此函数
        # 如果验证成功，生成 JWT 令牌
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": email}, expires_delta=access_token_expires
        )
        return {"message": "Verification successful", "access_token": access_token, "redirect_url": "/features?token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIyMTQwNzE3NjMyQHFxLmNvbSIsImV4cCI6MTc0NDg5NTk3OX0.Ji4jEIAaJiYpPj_OFYyS7rToXkp9SYoikj0gaTs6TPY"}
    else:
        raise HTTPException(status_code=400, detail="Invalid verification code")

@router.post("/scan")
async def scan(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # 确保文件保存到 UPLOADS_DIR 目录
        file_path = UPLOADS_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(contents)

        # 调用处理函数
        result = process_document_image(contents)  # 调用函数并赋值给 result

        # 确保返回的路径格式正确
        processed_path = result['processed_path']
        # 如果processed_path已经包含了'processed/'前缀，则不需要再添加
        if not processed_path.startswith('processed/'):
            processed_path = f"processed/{processed_path}"

        return JSONResponse(content={
            "original_image": f"/static/uploads/{file.filename}",
            "processed_image": f"/static/{processed_path}",
            "message": "Document processed successfully"
        })
    except Exception as e:
        logger.error(f"处理文档时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

app.include_router(router, prefix="/api")

@app.get("/scan-form")
async def scan_form(request: Request):
    return templates.TemplateResponse("scan_form.html", {"request": request})

@app.post("/api/scan")
async def api_scan(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # Save the file to the UPLOADS_DIR
        file_path = UPLOADS_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"上传文件保存到: {file_path}")
        
        # 检查上传文件是否存在
        if not file_path.exists():
            logger.error(f"上传文件不存在: {file_path}")
            return JSONResponse(status_code=500, content={"error": "上传文件保存失败"})

        # Call the processing function
        result = process_document_image(contents)

        # 确保返回的路径格式正确
        processed_path = result['processed_path']
        # 如果processed_path已经包含了'processed/'前缀，则不需要再添加
        if not processed_path.startswith('processed/'):
            processed_path = f"processed/{processed_path}"
            
        # 检查文件是否存在
        processed_file_path = Path(__file__).resolve().parent / "static" / processed_path
        logger.info(f"处理后文件路径: {processed_file_path}, 文件存在: {processed_file_path.exists()}")

        # Return the URL path
        return JSONResponse(content={
            "original_image": f"/static/uploads/{file.filename}",
            "processed_image": f"/static/{processed_path}",
            "message": "Document processed successfully"
        })

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        # Return a JSON response on error
        return JSONResponse(status_code=500, content={"error": f"Error processing document: {str(e)}"})

# 其他辅助函数
def generate_verification_code():
    # 生成随机验证码
    import random
    return str(random.randint(100000, 999999))

def store_verification_code(email, code):
    # 存储验证码，例如在数据库或者内存字典中
    # 这里使用一个简单的内存字典作为示例
    verification_codes[email] = code
    return True

@app.get("/features")
async def features_page(request: Request):
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    return templates.TemplateResponse("features.html", {"request": request})

def verify_user_code(email, code):
    # 验证用户提供的验证码
    return email in verification_codes and verification_codes[email] == code

def send_email(email, code):
    # 发送包含验证码的邮件
    # 使用 SMTP 配置从.env 文件
    # 略过实现，您应该有这部分代码
    return True

# 全局变量用于存储验证码（实际应用中应使用数据库）
verification_codes = {}

# 启动服务器时添加端口冲突处理
if __name__ == "__main__":
    import uvicorn
    # 启动Streamlit应用
    try:
        from app.streamlit_apps.launcher import main as start_apps
        import threading
        # 在后台线程中启动Streamlit应用
        threading.Thread(target=start_apps, daemon=True).start()
        print("Streamlit应用已启动")
    except Exception as e:
        print(f"启动Streamlit应用时出错: {str(e)}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)