import streamlit as st
import numpy as np
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time
import joblib
import os
from knn_classifier import KNNClassifier

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
                st.session_state.classifier = joblib.load(classifier_path)
                st.success("成功加载已保存的分类器!")
            else:
                st.session_state.classifier = KNNClassifier()
                st.info("未找到保存的分类器，已创建新分类器")
        except Exception as e:
            st.error(f"加载分类器时出错: {str(e)}")
            st.session_state.classifier = KNNClassifier()
            st.info("已创建新分类器")

    if'model' not in st.session_state:
        try:
            with st.spinner("正在加载MobileNet模型..."):
                model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
                st.session_state.model = torch.nn.Sequential(*list(model.children())[:-1])
                st.session_state.model.eval()
            st.success("模型加载完成!")
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            st.stop()

    # 初始化摄像头状态
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
        st.session_state.frame = None
        st.session_state.activation = None
        st.session_state.last_prediction = None
        st.session_state.last_confidence = 0.0
        st.session_state.cap = None

    # 创建布局
    col1, col2 = st.columns([3, 2])

    with col1:
        # 摄像头配置
        st.subheader("摄像头配置")
        camera_source = st.text_input(
            "摄像头源 (0=本地摄像头, 或输入网络摄像头URL)",
            value="0",
            help="输入0使用本地摄像头，或输入网络摄像头URL如: http://192.168.1.100:8080/video"
        )
        
        # 摄像头控制按钮
        if st.button(f"{'停止' if st.session_state.camera_on else '启动'}摄像头"):
            st.session_state.camera_on = not st.session_state.camera_on
            if st.session_state.camera_on:
                try:
                    # 尝试解析摄像头源
                    if camera_source.isdigit():
                        # 本地摄像头
                        st.session_state.cap = cv2.VideoCapture(int(camera_source))
                    else:
                        # 网络摄像头URL
                        st.session_state.cap = cv2.VideoCapture(camera_source)
                    
                    if not st.session_state.cap.isOpened():
                        st.error("无法访问摄像头，请检查权限或连接")
                        st.session_state.camera_on = False
                        st.session_state.cap = None
                    else:
                        st.success(f"成功连接到摄像头源: {camera_source}")
                except Exception as e:
                    st.error(f"启动摄像头时出错: {str(e)}")
                    st.session_state.camera_on = False
                    st.session_state.cap = None
            else:
                if st.session_state.cap is not None:
                    try:
                        st.session_state.cap.release()
                        st.session_state.cap = None
                        st.success("摄像头已停止")
                    except Exception as e:
                        st.error(f"停止摄像头时出现错误: {str(e)}")

        # 使用st.empty()作为视频帧占位符
        frame_placeholder = st.empty()

        # 预测结果显示区域
        prediction_placeholder = st.empty()

    with col2:
        # 样本计数显示
        st.subheader("训练样本数量")
        st.write(f"A类: {len(st.session_state.classifier.examples['A'])} 个样本")
        st.write(f"B类: {len(st.session_state.classifier.examples['B'])} 个样本")
        st.write(f"C类: {len(st.session_state.classifier.examples['C'])} 个样本")

        # 添加样本按钮
        st.subheader("添加训练样本")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            if st.button("添加A类样本", key="add_a"):
                if st.session_state.activation is not None:
                    count = st.session_state.classifier.add_example(st.session_state.activation, "A")
                    st.success(f"A类样本已添加 (总数: {count})")
                    try:
                        classifier_path = os.path.join(os.path.dirname(__file__), 'knn_classifier.joblib')
                        joblib.dump(st.session_state.classifier, classifier_path, compress=3)
                    except Exception as e:
                        st.error(f"保存分类器时出错: {str(e)}")
        with col_b:
            if st.button("添加B类样本", key="add_b"):
                if st.session_state.activation is not None:
                    count = st.session_state.classifier.add_example(st.session_state.activation, "B")
                    st.success(f"B类样本已添加 (总数: {count})")
                    try:
                        classifier_path = os.path.join(os.path.dirname(__file__), 'knn_classifier.joblib')
                        joblib.dump(st.session_state.classifier, classifier_path, compress=3)
                    except Exception as e:
                        st.error(f"保存分类器时出错: {str(e)}")
        with col_c:
            if st.button("添加C类样本", key="add_c"):
                if st.session_state.activation is not None:
                    count = st.session_state.classifier.add_example(st.session_state.activation, "C")
                    st.success(f"C类样本已添加 (总数: {count})")
                    try:
                        classifier_path = os.path.join(os.path.dirname(__file__), 'knn_classifier.joblib')
                        joblib.dump(st.session_state.classifier, classifier_path, compress=3)
                    except Exception as e:
                        st.error(f"保存分类器时出错: {str(e)}")

        if st.button("清除所有样本"):
            st.session_state.classifier = KNNClassifier()
            st.success("所有样本已清除")
            try:
                classifier_path = os.path.join(os.path.dirname(__file__), 'knn_classifier.joblib')
                joblib.dump(st.session_state.classifier, classifier_path, compress=3)
            except Exception as e:
                st.error(f"保存分类器时出错: {str(e)}")

    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 视频处理循环
    if st.session_state.camera_on and st.session_state.cap is not None:
        try:
            while st.session_state.camera_on:
                # 读取帧
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("无法读取摄像头画面")
                    st.session_state.camera_on = False
                    if st.session_state.cap is not None:
                        st.session_state.cap.release()
                        st.session_state.cap = None
                    break

                # 显示帧
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)

                # 特征提取
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    input_tensor = preprocess(pil_image)
                    input_batch = input_tensor.unsqueeze(0)

                    with torch.no_grad():
                        activation = st.session_state.model(input_batch)
                        activation = activation.squeeze().flatten().numpy()

                    st.session_state.activation = activation
                    st.session_state.frame = frame

                    # 预测
                    has_samples = any(len(v) > 0 for v in st.session_state.classifier.examples.values())
                    if has_samples:
                        pred_class, confidence = st.session_state.classifier.predict_class(activation)
                        st.session_state.last_prediction = pred_class
                        st.session_state.last_confidence = confidence

                        if pred_class:
                            prediction_placeholder.success(
                                f"预测结果: {pred_class}类 (置信度: {confidence:.2f})"
                            if confidence > 0.5 else
                                f"预测结果: {pred_class}类 (置信度较低: {confidence:.2f})"
                            )
                except Exception as e:
                    st.error(f"处理帧时出错: {str(e)}")

                # 控制帧率
                time.sleep(0.01)

        except Exception as e:
            st.error(f"处理错误: {str(e)}")
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
    else:
        frame_placeholder.info("摄像头已关闭，点击按钮启动")
        if st.session_state.last_prediction:
            prediction_placeholder.info(
                f"上次预测: {st.session_state.last_prediction}类 (置信度: {st.session_state.last_confidence:.2f})"
            )

    # 在页面关闭时释放摄像头资源
    if 'cap' in st.session_state and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None

    st.markdown("<div class='footer'>© 2025 机器学习算法平台 | wangxianfu.top</div>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()