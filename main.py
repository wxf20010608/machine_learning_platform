import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile, Depends
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.db import Base, engine
from app.email_utils import send_verification_email
from app.scan.processor import process_document_image
from app.auth.router import router as auth_router
from app.health import router as health_router
from app.scan.router import router as scan_router
from app.ml_routers import router as ml_router
from app.auth.service import create_access_token
import logging
from fastapi.middleware.cors import CORSMiddleware
from typing import Union

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
    """
    机器学习功能页面
    
    显示机器学习功能的入口页面，需要有效的JWT token才能访问。
    页面包含各种机器学习算法的链接和说明。
    
    - **token**: 查询参数，JWT认证token
    
    返回机器学习功能页面的HTML模板。
    """
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    return templates.TemplateResponse("features.html", {"request": request})

# 添加Streamlit应用路由
@app.get("/knn")
async def knn_app(request: Request):
    """
    KNN算法应用页面
    
    重定向到KNN (K-近邻) 算法的Streamlit应用页面。
    需要有效的JWT token才能访问。
    
    - **token**: 查询参数，JWT认证token
    
    返回重定向响应，跳转到KNN Streamlit应用。
    """
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    return RedirectResponse(url=f"/knn/?token={token}") # 通过Nginx代理，并直接重定向

@app.get("/kmeans")
async def kmeans_app(request: Request):
    """
    K-means聚类算法应用页面
    
    重定向到K-means聚类算法的Streamlit应用页面。
    需要有效的JWT token才能访问。
    
    - **token**: 查询参数，JWT认证token
    
    返回重定向响应，跳转到K-means Streamlit应用。
    """
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    return RedirectResponse(url=f"/kmeans/?token={token}")

@app.get("/linear-regression")
async def linear_regression_app(request: Request):
    """
    线性回归算法应用页面
    
    重定向到线性回归算法的Streamlit应用页面。
    需要有效的JWT token才能访问。
    
    - **token**: 查询参数，JWT认证token
    
    返回重定向响应，跳转到线性回归 Streamlit应用。
    """
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    return RedirectResponse(url=f"/linear/?token={token}")

@app.get("/logistic-regression")
async def logistic_regression_app(request: Request):
    """
    逻辑回归算法应用页面
    
    重定向到逻辑回归算法的Streamlit应用页面。
    需要有效的JWT token才能访问。
    
    - **token**: 查询参数，JWT认证token
    
    返回重定向响应，跳转到逻辑回归 Streamlit应用。
    """
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    return RedirectResponse(url=f"/logistic/?token={token}")

@app.get("/random-forest")
async def random_forest_app(request: Request):
    """
    随机森林算法应用页面
    
    重定向到随机森林算法的Streamlit应用页面。
    需要有效的JWT token才能访问。
    
    - **token**: 查询参数，JWT认证token
    
    返回重定向响应，跳转到随机森林 Streamlit应用。
    """
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    return RedirectResponse(url=f"/forest/?token={token}")

# 添加启动Streamlit应用的路由
@app.get("/start-streamlit-apps")
async def start_streamlit_apps():
    """
    启动所有Streamlit应用
    
    在后台线程中启动所有机器学习算法的Streamlit应用。
    包括KNN、K-means、线性回归、逻辑回归、随机森林等应用。
    
    返回启动状态信息。如果启动失败，返回错误详情。
    """
    try:
        from app.streamlit_apps.launcher import main as start_apps
        import threading
        # 在后台线程中启动Streamlit应用
        threading.Thread(target=start_apps, daemon=True).start()
        return {"message": "Streamlit应用已启动"}
    except Exception as e:
        logger.error(f"启动Streamlit应用时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"启动Streamlit应用失败: {str(e)}")


@app.get("/")
async def root(request: Request):
    """
    根路径 - 登录页面
    
    返回用户登录页面，用户可以通过邮箱验证码或GitHub OAuth进行登录。
    """
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/login")
async def login_page(request: Request):
    """
    登录页面
    
    显示用户登录界面，支持邮箱验证码和GitHub OAuth两种登录方式。
    """
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/scan")
async def scan_page(request: Request):
    """
    文档扫描页面
    
    显示文档扫描功能的页面，用户可以上传图片进行文档处理和分析。
    """
    return templates.TemplateResponse("scan.html", {"request": request})

@app.get("/scan/app", tags=["scan"])
async def scan_app(request: Request):
    """
    扫描应用页面
    
    提供文档扫描应用的主界面，包含上传、处理、预览等功能。
    """
    return templates.TemplateResponse("scan_app.html", {"request": request})

@app.get("/test-email")
async def test_email_page(request: Request):
    """
    邮件测试页面
    
    用于测试邮件发送功能的页面，开发调试时使用。
    """
    return templates.TemplateResponse("test_email.html", {"request": request})

# 发送验证码路由 - 使用数据库存储
@app.post("/send-verification-code/")
async def send_verification_code(email: str = Form(...)):
    logger.info(f"收到发送验证码请求: {email}")

    # 生成验证码
    verification_code = generate_verification_code()
    logger.info(f"生成验证码: {verification_code} 发送到: {email}")

    # 存储验证码到数据库
    from app.auth.service import store_verification_code
    from app.db import get_db
    
    db = next(get_db())
    try:
        store_verification_code(db, email, verification_code)
    except Exception as e:
        logger.error(f"存储验证码失败: {str(e)}")
        raise HTTPException(status_code=500, detail="存储验证码失败")
    finally:
        db.close()

    # 发送邮件
    result = send_verification_email(email, verification_code)

    if result:
        logger.info(f"成功发送验证码到 {email}")
        return {"message": "验证码已成功发送"}
    else:
        logger.error(f"发送验证码到 {email} 失败")
        raise HTTPException(status_code=500, detail="发送验证码失败，请检查邮箱地址是否正确")


# 验证码验证路由 - 使用数据库验证
@app.post("/verify-code/")
async def verify_code(email: str = Form(...), code: str = Form(...)):
    """
    验证邮箱验证码
    
    验证用户提供的邮箱和验证码，验证成功后生成JWT token。
    验证码有时效性，过期后需要重新获取。
    
    - **email**: 用户邮箱地址 (form-data格式)
    - **code**: 6位数字验证码 (form-data格式)
    
    验证成功后返回JWT token和重定向URL，失败返回错误信息。
    """
    # 验证用户提供的验证码
    from app.auth.service import verify_code as verify_code_db
    from app.db import get_db
    
    db = next(get_db())
    try:
        is_valid = verify_code_db(db, email, code)
    except Exception as e:
        logger.error(f"验证验证码失败: {str(e)}")
        raise HTTPException(status_code=500, detail="验证验证码失败")
    finally:
        db.close()
    
    if is_valid:
        # 如果验证成功，生成 JWT 令牌
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": email}, expires_delta=access_token_expires
        )
        # 使用动态生成的token构建redirect_url
        redirect_url = f"/features?token={access_token}"
        return {"message": "Verification successful", "access_token": access_token, "redirect_url": redirect_url}
    else:
        raise HTTPException(status_code=400, detail="验证码无效或已过期")

@router.post("/scan")
async def scan(file: UploadFile = File(...)):
    """
    文档扫描处理 (旧版API)
    
    上传图片文件进行文档扫描和图像处理。
    支持JPEG和PNG格式的图片文件。
    
    - **file**: 上传的图片文件
    
    返回原始图像和处理后图像的访问URL。
    """
    try:
        contents = await file.read()

        if file.filename is None:
            raise HTTPException(status_code=400, detail="文件名缺失")

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
    """
    扫描表单页面
    
    显示文档扫描的上传表单页面，用户可以通过表单上传图片进行扫描。
    """
    return templates.TemplateResponse("scan_form.html", {"request": request})

@app.post("/api/scan")
async def api_scan(file: UploadFile = File(...)):
    """
    文档扫描处理 (新版API)
    
    上传图片文件进行文档扫描和图像处理，包含更完善的错误处理。
    支持JPEG和PNG格式的图片文件。
    
    - **file**: 上传的图片文件 (multipart/form-data格式)
    
    返回处理结果，包含原始图像和处理后图像的访问URL。
    如果处理失败，返回错误信息。
    """
    try:
        contents = await file.read()

        if file.filename is None:
            raise HTTPException(status_code=400, detail="文件名缺失")

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
    """
    生成随机验证码
    
    生成6位数字的随机验证码，用于邮箱验证。
    
    返回:
        str: 6位数字验证码
    """
    # 生成随机验证码
    import random
    return str(random.randint(100000, 999999))

def store_verification_code(email, code):
    """
    存储验证码 (内存版本)
    
    将验证码存储到内存字典中。注意：生产环境应使用数据库存储。
    
    参数:
        email (str): 用户邮箱
        code (str): 验证码
    
    返回:
        bool: 存储是否成功
    """
    # 存储验证码，例如在数据库或者内存字典中
    # 这里使用一个简单的内存字典作为示例
    verification_codes[email] = code
    return True

@app.get("/features")
async def features_page(request: Request):
    """
    功能页面
    
    显示平台主要功能入口的页面，需要有效的JWT token才能访问。
    包含各种机器学习算法和文档处理功能的链接。
    
    - **token**: 查询参数，JWT认证token
    
    返回功能页面的HTML模板。
    """
    token = request.query_params.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")
    return templates.TemplateResponse("features.html", {"request": request})

def verify_user_code(email, code):
    """
    验证用户验证码 (内存版本)
    
    验证用户提供的邮箱和验证码是否匹配。
    注意：生产环境应使用数据库验证。
    
    参数:
        email (str): 用户邮箱
        code (str): 验证码
    
    返回:
        bool: 验证是否成功
    """
    # 验证用户提供的验证码
    return email in verification_codes and verification_codes[email] == code

def send_email(email, code):
    """
    发送邮件 (占位符函数)
    
    发送包含验证码的邮件。这是一个占位符函数，
    实际实现应使用SMTP配置从.env文件读取。
    
    参数:
        email (str): 目标邮箱
        code (str): 验证码
    
    返回:
        bool: 发送是否成功
    """
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