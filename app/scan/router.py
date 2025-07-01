from pathlib import Path
from datetime import datetime
import os

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer
import logging
from app.scan.processor import process_document_image
from app.auth.service import get_current_user

router = APIRouter(prefix="/api", tags=["scan"])
logger = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


@router.post("/scan")
async def scan_document(file: UploadFile = File(...)):
    try:
        # 验证文件类型
        allowed_types = ['image/jpeg', 'image/png', 'image/jpg']
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="只支持 JPEG 和 PNG 格式的图片")

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="上传的文件为空")

        # 获取基础目录
        base_dir = Path(__file__).resolve().parent.parent.parent
        static_dir = base_dir / "static"
        uploads_dir = static_dir / "uploads"
        processed_dir = static_dir / "processed"
        
        # 确保所有必要的目录都存在
        for directory in [static_dir, uploads_dir, processed_dir]:
            directory.mkdir(exist_ok=True, parents=True)
        
        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        original_filename = f"original_{timestamp}.jpg"
        original_path = uploads_dir / original_filename
        
        # 保存原始文件
        try:
            with open(original_path, "wb") as f:
                f.write(contents)
        except Exception as e:
            logger.error(f"保存原始文件失败: {str(e)}")
            raise HTTPException(status_code=500, detail="保存原始文件失败")
            
        logger.info(f"上传文件保存到: {original_path}")

        # 处理图像
        try:
            result = process_document_image(contents)
        except Exception as e:
            logger.error(f"图像处理失败: {str(e)}", exc_info=True)
            # 返回原始图像作为处理结果
            return {
                "original_image": f"/static/uploads/{original_filename}",
                "processed_image": f"/static/uploads/{original_filename}",
                "message": f"图像处理失败，显示原始图像: {str(e)}"
            }

        # 检查处理结果
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="处理结果格式不正确")
            
        processed_filename = result.get('processed_path')
        if not processed_filename:
            raise HTTPException(status_code=500, detail="处理后的图像路径为空")
            
        # 检查处理后的文件是否存在
        processed_path = processed_dir / processed_filename
        if not processed_path.exists():
            logger.error(f"处理后的文件不存在: {processed_path}")
            # 返回原始图像作为备选
            return {
                "original_image": f"/static/uploads/{original_filename}",
                "processed_image": f"/static/uploads/{original_filename}",
                "message": "处理后的文件未生成，显示原始图像"
            }
            
        # 返回正确的静态文件URL
        return {
            "original_image": f"/static/uploads/{original_filename}",
            "processed_image": f"/static/processed/{processed_filename}",
            "message": result.get("message", "处理成功")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"扫描错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理错误: {str(e)}")


@router.get("/image/{filename:path}")
async def get_image(filename: str):
    """获取图像文件的端点"""
    try:
        base_dir = Path(__file__).resolve().parent.parent.parent
        file_path = base_dir / "static" / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="图像文件不存在")
            
        return FileResponse(str(file_path))
    except Exception as e:
        logger.error(f"获取图像失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取图像失败: {str(e)}")