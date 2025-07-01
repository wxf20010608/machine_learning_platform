import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib


# 定义 PyTorch 模型
class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        st.write(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

    # 绘制训练损失曲线
    fig, ax = plt.subplots()
    ax.plot(range(1, epochs + 1), train_losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss over Epochs')
    st.pyplot(fig)


# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=load_iris().target_names)
    return accuracy, cm, report


# Streamlit 应用
st.title("鸢尾花种类识别应用")
st.markdown(
    "本应用使用 PyTorch 深度学习算法，根据输入的鸢尾花特征值预测其种类。你可以灵活选择训练轮数，多次训练模型以优化性能。")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 转换为 PyTorch 张量
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# 创建数据加载器
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# 模型初始化
model = IrisClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练设置
st.subheader("训练设置")
epochs = st.number_input("请输入训练轮数", min_value=1, value=10)
if st.button("开始训练"):
    st.subheader("训练过程")
    train_model(model, train_loader, criterion, optimizer, epochs)
    accuracy, cm, report = evaluate_model(model, test_loader)

    # 保存模型
    joblib.dump(model, 'iris_model.joblib')
    st.success(f"模型已成功保存为 iris_model.joblib，测试集准确率: {accuracy:.4f}")

    # 展示模型准确率
    st.subheader("模型评估指标")
    st.write(f"模型在测试集上的准确率为: {accuracy:.4f}")
    st.markdown("**准确率（Accuracy）**：是分类模型中最常用的评估指标，表示分类正确的样本数占总样本数的比例。")

    # 展示混淆矩阵
    st.subheader("混淆矩阵")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues, xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    st.pyplot(fig)
    st.markdown(
        "**混淆矩阵（Confusion Matrix）**：是一种可视化工具，用于展示模型在每个类别上的分类情况。行表示真实类别，列表示预测类别。通过混淆矩阵可以直观地看出模型在哪些类别上容易混淆。")

    # 展示分类报告
    st.subheader("分类报告")
    st.text(report)
    st.markdown(
        "**分类报告（Classification Report）**：包含了精确率（Precision）、召回率（Recall）、F1 值（F1-score）等详细的评估指标，分别从不同角度衡量了模型在每个类别上的性能。")

# 输入特征值
st.subheader("请输入鸢尾花的四个特征值，以预测其种类")
sepal_length = st.number_input("花萼长度 (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input("花萼宽度 (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("花瓣长度 (cm)", min_value=0.0, max_value=10.0, value=1.5)
petal_width = st.number_input("花瓣宽度 (cm)", min_value=0.0, max_value=10.0, value=0.2)

# 当用户点击预测按钮时进行预测
if st.button("预测"):
    try:
        # 加载模型
        model = joblib.load('iris_model.joblib')
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_features = scaler.transform(input_features)
        input_features = torch.FloatTensor(input_features)
        model.eval()
        with torch.no_grad():
            outputs = model(input_features)
            _, predicted = torch.max(outputs.data, 1)
            species = iris.target_names[predicted.item()]
        st.success(f"预测结果: 这是 {species} 鸢尾花。")
    except FileNotFoundError:
        st.error("请先训练模型！")

st.markdown("<div class='footer'>© 2025 机器学习算法平台 | wangxianfu.top</div>", unsafe_allow_html=True)


