import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont, ImageOps
import io
import joblib
import os
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import streamlit as st

# 设置matplotlib支持中文显示
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建保存模型的目录
MODEL_DIR = "streamlit/models/saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_digits_data():
    """加载手写数字数据集"""
    digits = datasets.load_digits()
    return digits


def train_digit_model(X, y, test_size=0.2, force_train=False):
    model_path = os.path.join(MODEL_DIR, "digit_model_pytorch.pth")
    scaler_path = os.path.join(MODEL_DIR, "digit_scaler_pytorch.joblib")

    if not force_train and os.path.exists(model_path) and os.path.exists(scaler_path):
        st.info("正在加载已保存的模型...")
        model = LogisticRegressionModel(X.shape[1], 10)
        state_dict = torch.load(model_path)
        # 映射键
        new_state_dict = {}
        for key, value in state_dict.items():
            if key == "fc.weight":
                new_state_dict["linear.weight"] = value
            elif key == "fc.bias":
                new_state_dict["linear.bias"] = value
            else:
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
        scaler = joblib.load(scaler_path)

        try:
            X_sample = X[:10]
            X_sample = scaler.transform(X_sample)
            X_sample = torch.tensor(X_sample, dtype=torch.float32)
            with torch.no_grad():
                _ = model(X_sample)
            st.success("模型加载成功！")

            X_scaled = scaler.transform(X)
            X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(X_scaled)
                _, predicted = torch.max(outputs, 1)
            y_pred = predicted.numpy()
            accuracy = accuracy_score(y, y_pred)

            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_test_scaled = scaler.transform(X_test)
            X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
            with torch.no_grad():
                outputs = model(X_test_scaled)
                _, y_test_pred = torch.max(outputs, 1)
            y_test_pred = y_test_pred.numpy()
            cm = confusion_matrix(y_test, y_test_pred)
            report = classification_report(y_test, y_test_pred, output_dict=True)

            return {
               'model': model,
               'scaler': scaler,
                'accuracy': accuracy,
                'confusion_matrix': cm,
               'report': report,
                'X_test': X_test_scaled,
                'y_test': y_test,
                'y_pred': y_test_pred
            }
        except Exception as e:
            st.warning(f"加载的模型无效，将重新训练: {e}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = LogisticRegressionModel(X_train.shape[1], 10)
    # 初始化权重
    nn.init.kaiming_uniform_(model.linear.weight, mode='fan_in', nonlinearity='relu')
    nn.init.zeros_(model.linear.bias)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.001)

    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        model.eval()
        outputs = model(X_test)
        _, y_test_pred = torch.max(outputs, 1)
    y_test_pred = y_test_pred.numpy()

    accuracy = accuracy_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, output_dict=True)

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    st.success(f"模型已保存至: {model_path}")

    return {
       'model': model,
       'scaler': scaler,
        'accuracy': accuracy,
        'confusion_matrix': cm,
       'report': report,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_test_pred
    }


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        nn.init.kaiming_uniform_(self.linear.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        out = self.linear(x)
        return out


def preprocess_canvas_image(img_data, target_size=(8, 8)):
    img = Image.fromarray(img_data).convert('L')
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    img_array = 255 - img_array
    img_array = img_array / 255.0 * 16.0
    return img_array


def predict_digit(model, scaler, image_array):
    features = image_array.reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_scaled = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        model.eval()
        outputs = model(features_scaled)
        _, prediction = torch.max(outputs, 1)
        proba = torch.softmax(outputs, dim=1).numpy()[0]

    return prediction.item(), proba


def render_digit_recognition_page():
    st.markdown("<h1 style='text-align: center;'>手写数字识别</h1>", unsafe_allow_html=True)

    if st.button("返回首页", key="back_to_home"):
        st.session_state.current_page = "home"
        st.rerun()

    retrain = st.sidebar.checkbox("重新训练模型", value=False, help="勾选此项将强制重新训练模型，不使用保存的模型")

    digits = load_digits_data()

    st.subheader("数据集示例")
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(digits.images[i], cmap='binary')
        ax.set_title(f"标签: {digits.target[i]}")
        ax.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

    st.write(f"数据集大小: {digits.data.shape[0]}个样本, 每个样本{digits.data.shape[1]}个特征")

    # 仅在需要重新训练或模型不存在时进行训练
    if retrain or (not os.path.exists(os.path.join(MODEL_DIR, "digit_model_pytorch.pth")) or
                   not os.path.exists(os.path.join(MODEL_DIR, "digit_scaler_pytorch.joblib"))):
        with st.spinner("模型训练中..."):
            result = train_digit_model(digits.data, digits.target, force_train=retrain)
            model = result['model']
            scaler = result['scaler']
            accuracy = result['accuracy']
            cm = result['confusion_matrix']
            report = result['report']
    else:
        result = train_digit_model(digits.data, digits.target, force_train=retrain)
        model = result['model']
        scaler = result['scaler']
        accuracy = result['accuracy']
        cm = result['confusion_matrix']
        report = result['report']

    st.subheader("模型性能")
    st.write(f"模型准确率: {accuracy:.4f}")

    st.write("混淆矩阵:")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    st.pyplot(fig)

    st.subheader("绘制数字进行识别")
    st.write("在下方画板上绘制一个数字(0 - 9)，然后点击识别按钮:")

    if 'canvas_id' not in st.session_state:
        st.session_state.canvas_id = 0

    col1, col2 = st.columns([2, 1])

    with col1:
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_id}"
        )

        if st.button("清除画布"):
            st.session_state.canvas_id += 1
            st.rerun()

    preprocessed_img = None
    if canvas_result.image_data is not None:
        preprocessed_img = preprocess_canvas_image(canvas_result.image_data)

        with col2:
            st.write("处理后的图像:")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(preprocessed_img, cmap='binary')
            ax.axis('off')
            st.pyplot(fig)

    if st.button("识别数字", type="primary"):
        if preprocessed_img is not None:
            prediction, probas = predict_digit(model, scaler, preprocessed_img)

            st.success(f"预测结果: {prediction}")

            st.write("各数字的概率:")
            fig, ax = plt.subplots(figsize=(10, 4))
            bars = ax.bar(range(10), probas)
            bars[prediction].set_color('red')

            plt.xticks(range(10))
            plt.xlabel('数字')
            plt.ylabel('概率')
            plt.title('预测概率分布')
            st.pyplot(fig)
        else:
            st.warning("请先在画布上绘制一个数字")

    st.info("提示：尽量画出清晰的数字，使数字大小适中并居中。如果识别不准确，可以尝试重新绘制。")
    st.markdown("<div class='footer'>© 2025 机器学习算法平台 | wangxianfu.top</div>", unsafe_allow_html=True)



if __name__ == "__main__":
    render_digit_recognition_page()