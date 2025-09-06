# type: ignore
import cv2
import numpy as np
from datetime import datetime
import os
import io
from PIL import Image
import base64
import logging
from pathlib import Path
# 移除pdb调试模块

# 配置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


def preprocess_image_for_detection(image, max_height=1500):
    """图像预处理，用于边缘检测"""
    # 调整图像大小保持纵横比
    height, width = image.shape[:2]
    if height > max_height:
        ratio = max_height / height
        image = cv2.resize(image, (int(width * ratio), max_height))
    elif width > max_height:  # 添加宽度限制
        ratio = max_height / width
        image = cv2.resize(image, (max_height, int(height * ratio)))

    # 创建副本并处理
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 使用卷积核进行锐化操作
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(blurred, -1, kernel)

    return orig, gray, sharpened


def detect_edges_with_enhancement(sharpened):
    """增强的边缘检测"""
    # 使用Canny算子获取边缘
    edges = cv2.Canny(sharpened, 50, 150)
    
    # 使用形态学操作清理边缘
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    return edges


def find_best_document_contour(edges, orig):
    height, width = orig.shape[:2]
    # 根据图像分辨率自适应调整面积过滤阈值
    area_threshold = 0.03
    if width * height > 500000:  # 假设图像像素总数大于500000时
        area_threshold = 0.05
    elif width * height < 200000:  # 假设图像像素总数小于200000时
        area_threshold = 0.01

    # 寻找轮廓
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("contours shape:", [c.shape for c in contours])  # 添加调试信息

    # 确保 contours 是有效的轮廓列表
    valid_contours = []
    for contour in contours:
        if isinstance(contour, np.ndarray) and contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2:
            valid_contours.append(contour)
    contours = valid_contours

    # 按面积从大到小排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 过滤掉面积过小的轮廓
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > area_threshold * (width * height)]
    print("valid_contours shape:", [c.shape for c in valid_contours])  # 添加调试信息

    # 使用霍夫直线变换检测直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        line_points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_points.extend([[x1, y1], [x2, y2]])
        line_points = np.array(line_points, dtype=np.int32)
        hull = cv2.convexHull(line_points)
        valid_contours.append(hull)

    doc_contour = None
    for contour in valid_contours:
        perimeter = cv2.arcLength(contour, True)
        # 根据图像分辨率自适应调整拟合精度
        if width * height > 500000:  # 图像较大时
            approx = cv2.approxPolyDP(contour, 0.005 * perimeter, True)
        elif width * height < 200000:  # 图像较小时
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        else:
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        print("approx shape:", approx.shape if approx is not None else None)  # 添加调试信息

        # 计算轮廓紧凑度（周长的平方与面积的比值），排除紧凑度异常大的轮廓（可能是噪声干扰）
        compactness = (perimeter ** 2) / (cv2.contourArea(approx) + 1e-8)
        # 计算轮廓的长宽比，排除长宽比异常的轮廓（纸张长宽比一般有一定范围）
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h if h != 0 else 0

        if len(approx) > 0 and len(approx) >= 4:
            try:
                hull = cv2.convexHull(approx)
                if len(hull) > 0:
                    defects = cv2.convexityDefects(approx, hull)
                    if defects is not None and len(defects) > 0:
                        # 计算缺陷程度，这里简单以缺陷数量占轮廓点数比例判断
                        defect_ratio = len(defects) / len(approx)
                        if defect_ratio > 0.1:  # 假设缺陷比例大于0.1则排除
                            continue
            except cv2.error as e:
                continue

        if len(approx) == 4 and compactness < 100 and 0.5 < aspect_ratio < 2:
            doc_contour = approx
            break

    if doc_contour is None and valid_contours:
        hull = cv2.convexHull(valid_contours[0])
        perimeter = cv2.arcLength(hull, True)
        doc_contour = cv2.approxPolyDP(hull, 0.02 * perimeter, True)

        if len(doc_contour) != 4:
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            doc_contour = np.int32(box)

    # 坐标校准：检查角点是否在图像内且合理，若不在则调整
    for i in range(len(doc_contour)):
        try:
            # 检查doc_contour[i]的结构
            if isinstance(doc_contour[i], np.ndarray):
                # 如果是一个形状为(1,2)的数组，需要正确访问x和y
                if doc_contour[i].shape[0] == 1 and doc_contour[i].shape[1] == 2:
                    x, y = doc_contour[i][0][0], doc_contour[i][0][1]
                elif doc_contour[i].size >= 2:  # 如果是一维数组且至少有两个元素
                    x, y = doc_contour[i][0], doc_contour[i][1]
                else:
                    print(f"无法解析的轮廓点格式: {doc_contour[i]}, 形状: {doc_contour[i].shape}")
                    x, y = 0, 0
            else:
                print(f"非数组类型的轮廓点: {type(doc_contour[i])}")
                x, y = 0, 0
        except Exception as e:
            print(f"处理轮廓点时出错: {str(e)}, 索引: {i}, 值: {doc_contour[i]}")
            x, y = 0, 0
            
        # 确保坐标在图像范围内
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        
        # 根据doc_contour的结构更新坐标
        try:
            if isinstance(doc_contour[i], np.ndarray):
                if doc_contour[i].shape[0] == 1 and doc_contour[i].shape[1] == 2:
                    doc_contour[i][0][0] = x
                    doc_contour[i][0][1] = y
                elif doc_contour[i].size >= 2:
                    doc_contour[i][0] = x
                    doc_contour[i][1] = y
        except Exception as e:
            print(f"更新轮廓点时出错: {str(e)}, 索引: {i}, 值: {doc_contour[i]}")

    if doc_contour is None or len(doc_contour) < 4:
        doc_contour = np.array([
            [[0, 0]], [[width - 1, 0]],
            [[width - 1, height - 1]], [[0, height - 1]]
        ], dtype=np.int32)

    doc_contour = doc_contour.reshape(4, 2)
    return doc_contour


def safe_affine_transform(image, pts):
    """安全的仿射变换，包含错误处理"""
    if pts is None or len(pts) < 3:
        raise ValueError("仿射变换需要至少3个点")

    try:
        # 更精确地选取三个点（这里先对轮廓点进行排序，再选取左上、右上、左下）
        rect = order_points(pts)
        src_pts = np.float32([rect[0], rect[1], rect[3]])
        height, width = image.shape[:2]
        dst_pts = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])

        # 计算仿射变换矩阵
        M = cv2.getAffineTransform(src_pts, dst_pts)

        # 应用仿射变换
        warped = cv2.warpAffine(image, M, (width, height),
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_REPLICATE)

        return warped
    except Exception as e:
        logger.error(f"仿射变换失败: {str(e)}")
        return None


def enhance_document_quality(image):
    """高质量文档增强"""
    try:
        # 转为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 增加高斯模糊平滑处理
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # 调整clipLimit参数
        enhanced = clahe.apply(gray)

        # 非局部均值去噪
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        # 自适应二值化
        binary = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 5  # 调整参数
        )

        # 转换为彩色输出
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        logger.error(f"图像增强失败: {str(e)}")
        return image


def order_points(pts):
    """将四个点按左上、右上、右下、左下的顺序排列"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def save_image(image, filename):
    """保存图像到指定文件并确保正确保存"""
    try:
        # 使用绝对路径，确保文件保存在正确的位置
        base_dir = Path(__file__).resolve().parent.parent.parent
        output_dir = base_dir / "static" / "processed"
        output_dir.mkdir(exist_ok=True, parents=True)
        file_path = output_dir / filename
        
        logger.info(f"保存图像到路径: {file_path}")

        # 确保图像是有效的
        if image is None or image.size == 0:
            raise ValueError("无效的图像数据")

        # 压缩图像以减少文件大小
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]  # 降低质量以加快保存速度
        success, img_buf = cv2.imencode('.jpg', image, encode_params)
        if not success:
            raise ValueError("图像编码失败")

        # 直接写入文件
        with open(file_path, 'wb') as f:
            f.write(img_buf.tobytes())

        return filename
    except Exception as e:
        logger.error(f"保存图像时出错: {str(e)}")
        return None

def get_image_base64(image_path):
    """将图像文件转换为base64编码，用于HTML直接显示"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"转换图像为b64时出错: {str(e)}")
        return None


def pil_to_bytes(pil_image, format='JPEG'):
    """将PIL图像转换为字节流"""
    try:
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format=format, quality=95)
        return img_byte_arr.getvalue()
    except Exception as e:
        logger.error(f"转换PIL图像为字节流时出错: {str(e)}")
        return None


def cv2_to_pil(cv2_image):
    """将OpenCV图像转换为PIL图像"""
    try:
        # 从BGR转换为RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    except Exception as e:
        logger.error(f"转换OpenCV图像为PIL图像时出错: {str(e)}")
        return None


def crop_image(image, contour):
    """根据轮廓裁剪图像"""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = box.astype(int)

    # 找到最小外接矩形的顶点坐标
    x_min = min(box[:, 0])
    x_max = max(box[:, 0])
    y_min = min(box[:, 1])
    y_max = max(box[:, 1])

    # 裁剪图像
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


def process_document_image(image_bytes):
    """处理文档图像"""
    try:
        # 将字节转换为numpy数组
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("无法解码图像数据")

        # 限制图像最大尺寸以提高处理速度
        max_dimension = 1500
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            image = cv2.resize(image, None, fx=scale, fy=scale)

        # 图像预处理
        orig, gray, sharpened = preprocess_image_for_detection(image)
        
        # 边缘检测
        edges = detect_edges_with_enhancement(sharpened)
        
        # 查找文档轮廓
        doc_contour = find_best_document_contour(edges, orig)
        
        if doc_contour is None or len(doc_contour) != 4:
            logger.warning("未能找到有效的文档轮廓，将使用原始图像")
            processed_image = enhance_document_quality(orig)
        else:
            # 透视变换
            processed_image = crop_image(orig, doc_contour)
            if processed_image is None or processed_image.size == 0:
                logger.warning("透视变换失败，将使用原始图像")
                processed_image = enhance_document_quality(orig)
            else:
                # 增强处理后的图像
                processed_image = enhance_document_quality(processed_image)

        # 生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        processed_filename = f"processed_{timestamp}.jpg"
        
        # 保存处理后的图像
        processed_path = save_image(processed_image, processed_filename)
        
        if not processed_path:
            raise ValueError("保存处理后的图像失败")
            
        return {
            "processed_path": processed_filename,
            "message": "文档扫描处理成功"
        }
        
    except Exception as e:
        logger.error(f"处理文档图像时出错: {str(e)}", exc_info=True)
        raise ValueError(f"处理文档图像时出错: {str(e)}")


def process_image_and_convert_to_bytes(image_bytes):
    """处理图像并返回字节流，增加错误处理"""
    try:
        # 处理文档图像
        result = process_document_image(image_bytes)

        # 读取处理后的图像
        processed_image = cv2.imread(result["processed_path"])

        if processed_image is None:
            raise ValueError(f"无法读取处理后的图像文件: {result['processed_path']}")

        # 转换为PIL图像
        pil_image = cv2_to_pil(processed_image)

        if pil_image is None:
            raise ValueError("无法转换为PIL图像")

        # 转换为字节流
        processed_bytes = pil_to_bytes(pil_image)

        if processed_bytes is None or len(processed_bytes) == 0:
            raise ValueError("无法转换为字节流或字节流为空")

        return processed_bytes
    except Exception as e:
        logger.error(f"处理图像并转换为字节流时出错: {str(e)}")
        # 在失败的情况下返回原始图像字节
        return image_bytes
