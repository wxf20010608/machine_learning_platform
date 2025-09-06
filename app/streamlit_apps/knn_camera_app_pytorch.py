# type: ignore
import streamlit as st
import numpy as np
try:
    import cv2
except ImportError:
    st.error("OpenCV未安装，请运行: pip install opencv-python")
    st.stop()
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
import joblib
import os
from knn_classifier import KNNClassifier

def save_classifier(classifier):
    """保存分类器到文件"""
    try:
        classifier_path = os.path.join(os.path.dirname(__file__), 'knn_classifier.joblib')
        joblib.dump(classifier, classifier_path, compress=3)
        st.success("分类器已保存!")
    except Exception as e:
        st.error(f"保存分类器时出错: {str(e)}")
        st.warning("分类器将仅在当前会话中可用")

def main():
    st.set_page_config(page_title="KNN摄像头识别", layout="wide")

    # 页面标题和说明
    st.title("KNN摄像头实时识别 (PyTorch版)")
    st.markdown("""
    ### 使用说明:
    1. **摄像头配置**: 
       - 输入 `0` 使用本地摄像头（需要摄像头权限）
       - 输入网络摄像头URL，如 `http://192.168.1.100:8080/video`
       - 或者使用IP摄像头应用提供的URL
    2. 点击 **启动摄像头** 按钮开始捕获视频
    3. 使用下方按钮添加训练样本:
       - 按 **添加A类样本** 将当前画面添加到A类
       - 按 **添加B类样本** 将当前画面添加到B类
       - 按 **添加C类样本** 将当前画面添加到C类
    4. 添加足够的样本后，系统会自动预测当前画面属于哪个类别
    
    **注意**: 在Docker容器中，本地摄像头可能无法访问，建议使用网络摄像头URL。
    """)

    # 初始化会话状态
    if 'classifier' not in st.session_state:
        classifier_path = os.path.join(os.path.dirname(__file__), 'knn_classifier.joblib')
        try:
            if os.path.exists(classifier_path):
                # 尝试加载模型，如果失败则创建新的
                try:
                    st.session_state.classifier = joblib.load(classifier_path)
                    st.success("成功加载已保存的分类器!")
                except Exception as load_error:
                    st.warning(f"加载已保存的分类器失败: {str(load_error)}")
                    st.info("正在创建新的分类器...")
                    st.session_state.classifier = KNNClassifier()
                    # 尝试保存新的分类器
                    try:
                        joblib.dump(st.session_state.classifier, classifier_path, compress=3)
                        st.success("新分类器已创建并保存!")
                    except Exception as save_error:
                        st.warning(f"保存新分类器失败: {str(save_error)}")
            else:
                st.session_state.classifier = KNNClassifier()
                st.info("未找到保存的分类器，已创建新分类器")
                # 尝试保存新的分类器
                try:
                    joblib.dump(st.session_state.classifier, classifier_path, compress=3)
                    st.success("新分类器已保存!")
                except Exception as save_error:
                    st.warning(f"保存新分类器失败: {str(save_error)}")
        except Exception as e:
            st.error(f"初始化分类器时出错: {str(e)}")
            st.session_state.classifier = KNNClassifier()
            st.info("已创建新分类器")

    if 'model' not in st.session_state:
        try:
            with st.spinner("正在加载MobileNet模型..."):
                # 尝试多种方式加载模型
                model = None
                load_method = "unknown"
                
                # 直接使用未训练模型，避免网络下载问题
                try:
                    st.info("正在加载MobileNet模型（未训练版本）...")
                    model = models.mobilenet_v2(weights=None)
                    load_method = "untrained"
                    st.success("✓ 模型加载成功!")
                    st.info("注意：使用未训练模型，特征提取效果可能不如预训练模型，但足够用于KNN分类")
                except Exception as e:
                    st.error(f"模型加载失败: {str(e)}")
                    st.error("请检查PyTorch安装")
                    st.stop()
                
                # 创建特征提取器
                if model is not None:
                    st.session_state.model = torch.nn.Sequential(*list(model.children())[:-1])
                    st.session_state.model.eval()
                    st.session_state.model_load_method = load_method
                    st.success(f"模型初始化完成! (使用{load_method}模式)")
                else:
                    st.error("模型创建失败")
                    st.stop()
                    
        except Exception as e:
            st.error(f"模型加载过程中出现未知错误: {str(e)}")
            st.error("请检查PyTorch安装或重新启动应用")
            st.stop()

    # 初始化摄像头状态
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
        st.session_state.frame = None
        
    # 添加页面刷新检测
    if 'page_loaded' not in st.session_state:
        st.session_state.page_loaded = True
        # 初始化rerun标记
        st.session_state.rerun_triggered = False
    else:
        # 页面已经加载过，这是一次刷新
        # 只有在非rerun触发的刷新且摄像头已关闭的情况下才释放资源
        if not hasattr(st.session_state, 'rerun_triggered') or not st.session_state.rerun_triggered:
            if not st.session_state.camera_on and 'cap' in st.session_state and st.session_state.cap is not None:
                try:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                except:
                    pass
                
    # 重置激活状态，但保留预测结果
    if not st.session_state.camera_on:
        st.session_state.activation = None

    # 创建三列布局，优化空间利用
    col1, col2, col3 = st.columns([3, 2, 2])

    with col1:
        # 摄像头配置
        st.subheader("📹 摄像头控制")
        camera_source = st.text_input(
            "摄像头源 (0=本地摄像头, 或输入网络摄像头URL)",
            value="0",
            help="输入0使用本地摄像头，或输入网络摄像头URL如: http://192.168.1.100:8080/video"
        )
        
        # 摄像头控制按钮
        button_text = "⏹️ 停止摄像头" if st.session_state.camera_on else "🎥 启动摄像头"
        if st.button(button_text, use_container_width=True):
            # 切换摄像头状态
            new_camera_state = not st.session_state.camera_on
            
            # 先确保释放之前的摄像头资源
            if st.session_state.cap is not None:
                try:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                except Exception as e:
                    st.warning(f"释放摄像头资源时出现警告: {str(e)}")
            
            # 更新摄像头状态
            st.session_state.camera_on = new_camera_state
            
            # 如果是打开摄像头
            if st.session_state.camera_on:
                try:
                    # 尝试解析摄像头源
                    if camera_source.isdigit():
                        # 本地摄像头
                        st.session_state.cap = cv2.VideoCapture(int(camera_source))  # type: ignore
                    else:
                        # 网络摄像头URL
                        st.session_state.cap = cv2.VideoCapture(camera_source)  # type: ignore
                    
                    # 设置摄像头属性，提高性能
                    if st.session_state.cap is not None:
                        st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # type: ignore
                        st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # type: ignore
                    
                    if not st.session_state.cap.isOpened():
                        st.error("无法访问摄像头，请检查权限或连接")
                        st.session_state.camera_on = False
                        st.session_state.cap = None
                    else:
                        st.success(f"✅ 成功连接到摄像头源: {camera_source}")
                except Exception as e:
                    st.error(f"❌ 启动摄像头时出错: {str(e)}")
                    st.session_state.camera_on = False
                    st.session_state.cap = None
            else:
                # 关闭摄像头
                st.success("✅ 摄像头已停止")

        # 使用st.empty()作为视频帧占位符
        frame_placeholder = st.empty()

        # 预测结果显示区域
        prediction_placeholder = st.empty()

    with col2:
        # 训练样本控制
        st.subheader("🎯 训练样本")
        
        # 添加样本按钮 - 垂直排列，更大的按钮
        if st.button("➕ 添加A类样本", key="add_a", use_container_width=True):
            if st.session_state.activation is not None:
                count = st.session_state.classifier.add_example(st.session_state.activation, "A")
                st.success(f"✅ A类样本已添加 (总数: {count})")
                save_classifier(st.session_state.classifier)
            else:
                st.warning("⚠️ 请先启动摄像头")
        
        if st.button("➕ 添加B类样本", key="add_b", use_container_width=True):
            if st.session_state.activation is not None:
                count = st.session_state.classifier.add_example(st.session_state.activation, "B")
                st.success(f"✅ B类样本已添加 (总数: {count})")
                save_classifier(st.session_state.classifier)
            else:
                st.warning("⚠️ 请先启动摄像头")
        
        if st.button("➕ 添加C类样本", key="add_c", use_container_width=True):
            if st.session_state.activation is not None:
                count = st.session_state.classifier.add_example(st.session_state.activation, "C")
                st.success(f"✅ C类样本已添加 (总数: {count})")
                save_classifier(st.session_state.classifier)
            else:
                st.warning("⚠️ 请先启动摄像头")

        st.markdown("---")
        
        if st.button("🗑️ 清除所有样本", use_container_width=True):
            st.session_state.classifier = KNNClassifier()
            st.success("✅ 所有样本已清除")
            save_classifier(st.session_state.classifier)

    with col3:
        # 样本计数显示
        st.subheader("📊 样本统计")
        
        # 使用指示器显示样本数量
        a_count = len(st.session_state.classifier.examples['A'])
        b_count = len(st.session_state.classifier.examples['B'])
        c_count = len(st.session_state.classifier.examples['C'])
        
        col_a_stat, col_b_stat, col_c_stat = st.columns(3)
        with col_a_stat:
            st.metric("A类", a_count)
        with col_b_stat:
            st.metric("B类", b_count)
        with col_c_stat:
            st.metric("C类", c_count)
            
        # 模型状态
        st.subheader("🤖 模型状态")
        total_samples = sum(len(v) for v in st.session_state.classifier.examples.values())
        st.metric("总样本数", total_samples)
        st.metric("K值", st.session_state.classifier.n_neighbors)
        
        # 训练状态指示器
        if st.session_state.classifier.is_trained:
            st.success("✅ 模型已训练")
        else:
            st.warning("⚠️ 模型未训练")
        
        # 调试信息
        if hasattr(st.session_state, 'debug_info'):
            st.subheader("🔍 调试信息")
            st.text(st.session_state.debug_info)
            
        # 添加详细的样本信息
        st.subheader("📋 样本详情")
        for class_name in ['A', 'B', 'C']:
            samples = st.session_state.classifier.examples[class_name]
            if samples:
                st.text(f"{class_name}类样本:")
                for i, sample in enumerate(samples[:3]):  # 只显示前3个样本
                    st.text(f"  样本{i+1}: 范围[{sample.min():.4f}, {sample.max():.4f}], 均值{sample.mean():.4f}")
                if len(samples) > 3:
                    st.text(f"  ... 还有{len(samples)-3}个样本")

    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 视频处理 - 使用Streamlit的自动重新运行机制
    if st.session_state.camera_on and st.session_state.cap is not None:
        # 初始化帧处理计数器，用于控制刷新率
        if 'frame_counter' not in st.session_state:
            st.session_state.frame_counter = 0
        
        # 读取当前帧，最多尝试3次
        max_attempts = 3
        ret = False
        frame = None
        
        for attempt in range(max_attempts):
            if st.session_state.cap is not None:
                ret, frame = st.session_state.cap.read()
                if ret:
                    break
                time.sleep(0.1)  # 短暂等待后重试
        
        if not ret:
            st.error("无法读取摄像头画面，请尝试重新启动摄像头")
            # 不要自动关闭摄像头，让用户决定是否关闭
            # 只是显示错误信息，但保持摄像头状态
            frame_placeholder.error("⚠️ 摄像头连接中断，请点击停止按钮后重新启动")
            time.sleep(0.5)
            st.rerun()  # 重新运行以尝试恢复
        else:
            # 显示帧
            # 兼容不同Streamlit版本
            try:
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            except TypeError:
                # 旧版本不支持use_container_width参数
                frame_placeholder.image(frame, channels="BGR")
            
            # 特征提取和预测 - 每3帧执行一次以减少计算负担
            st.session_state.frame_counter += 1
            if st.session_state.frame_counter % 3 == 0:  # 每3帧处理一次
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
                    pil_image = Image.fromarray(frame_rgb)
                    input_tensor = preprocess(pil_image)
                    input_batch = input_tensor.unsqueeze(0)  # type: ignore

                    with torch.no_grad():
                        activation = st.session_state.model(input_batch)
                        activation = activation.squeeze().flatten().numpy()

                    st.session_state.activation = activation
                    st.session_state.frame = frame

                    # 预测
                    has_samples = any(len(v) > 0 for v in st.session_state.classifier.examples.values())
                    if has_samples:
                        try:
                            # 添加调试信息
                            st.session_state.debug_info = f"特征维度: {activation.shape}, 特征范围: [{activation.min():.4f}, {activation.max():.4f}]"
                            
                            pred_class, confidence = st.session_state.classifier.predict_class(activation)
                            st.session_state.last_prediction = pred_class
                            st.session_state.last_confidence = confidence

                            if pred_class:
                                # 显示预测结果，使用更好的UI
                                if confidence > 0.7:
                                    prediction_placeholder.success(
                                        f"🎯 **预测结果: {pred_class}类** (置信度: {confidence:.2f})"
                                    )
                                elif confidence > 0.4:
                                    prediction_placeholder.warning(
                                        f"⚠️ **预测结果: {pred_class}类** (置信度中等: {confidence:.2f})"
                                    )
                                else:
                                    prediction_placeholder.error(
                                        f"❓ **预测结果: {pred_class}类** (置信度较低: {confidence:.2f})"
                                    )
                            else:
                                prediction_placeholder.info("🔍 等待预测结果...")
                        except Exception as pred_error:
                            st.error(f"预测时出错: {str(pred_error)}")
                            st.error(f"错误详情: {type(pred_error).__name__}")
                except Exception as e:
                    st.error(f"处理帧时出错: {str(e)}")
                    st.warning("请检查模型是否正确加载")
            
            # 添加自动刷新机制
            if st.session_state.camera_on:
                # 标记这次rerun是由摄像头触发的
                st.session_state.rerun_triggered = True
                time.sleep(0.03)  # 控制帧率
                st.rerun()  # 重新运行应用以获取下一帧
    else:
        frame_placeholder.info("摄像头已关闭，点击按钮启动")
        if st.session_state.last_prediction:
            prediction_placeholder.info(
                f"上次预测: {st.session_state.last_prediction}类 (置信度: {st.session_state.last_confidence:.2f})"
            )

    # 不要在这里释放摄像头资源，否则会导致自动关闭
    # 资源释放应该只在用户明确关闭摄像头或页面真正关闭时进行

    st.markdown("<div class='footer'>© 2025 机器学习算法平台 | wangxianfu.top</div>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()