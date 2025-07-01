import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.font_manager import FontProperties
import matplotlib
import joblib
import os  # 导入os模块

# 设置支持中文的字体
try:
    # 设置中文显示
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei'] + plt.rcParams[
        'font.sans-serif']
    # 正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 创建保存模型的目录
MODEL_DIR = "streamlit/models/saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)


def load_california_data():
    """加载加利福尼亚州房价数据集"""
    housing = fetch_california_housing()
    data = pd.DataFrame(housing.data, columns=housing.feature_names)
    data['PRICE'] = housing.target
    return data, housing.feature_names


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        # 初始化权重
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


def train_model(data, test_size=0.2, epochs=1000, learning_rate=0.001, force_train=False):
    MODEL_PATH = os.path.join(MODEL_DIR, 'model.joblib')
    SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
    if not force_train and os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        st.info("正在加载已保存的模型...")
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # 从数据中提取特征和目标变量
        X = data.drop('PRICE', axis=1).values
        y = data['PRICE'].values
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        # 数据归一化
        X_test = scaler.transform(X_test)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    else:
        scaler = StandardScaler()
        X = data.drop('PRICE', axis=1).values
        y = data['PRICE'].values

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # 数据归一化
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        model = LinearRegressionModel(X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 保存模型和缩放器
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        st.success("模型已保存！")

    # 预测
    with torch.no_grad():
        y_pred = model(X_test)

    # 计算评估指标
    mse = nn.MSELoss()(y_pred, y_test).item()
    rmse = np.sqrt(mse)
    r2 = 1 - (mse / np.var(y_test.numpy()))

    # 获取特征重要性（系数的绝对值）
    importance = np.abs(model.linear.weight.detach().numpy()[0])
    importance = importance / np.sum(importance)

    # 准备结果
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

    feature_importance = dict(zip(data.drop('PRICE', axis=1).columns, importance))

    return {
       'model': model,
       'metrics': metrics,
        'feature_importance': feature_importance,
       'scaler': scaler
    }


def predict_housing_price(model, features, scaler):
    """使用模型预测房价"""
    features = scaler.transform([features])
    features = torch.tensor(features, dtype=torch.float32).view(1, -1)
    with torch.no_grad():
        return model(features).item()


def render_california_housing_page():
    """渲染加利福尼亚州房价预测页面"""
    st.markdown("<h1 class='header'>加利福尼亚州房价预测</h1>", unsafe_allow_html=True)

    # 项目介绍
    st.subheader("项目介绍")
    st.write("""
    加利福尼亚州房价预测是一个经典的回归问题。该数据集包含了加州各地区的房屋价格以及相关的特征变量。
    在这个示例中，我们使用线性回归模型来预测房屋价格，并分析影响房价的主要因素。
    """)

    # 训练参数设置
    epochs = st.slider("训练轮数", 100, 2000, 1000)
    learning_rate = 0.001  # 固定学习率
    force_train = st.checkbox("强制重新训练模型", value=False)

    # 加载数据
    data, feature_names = load_california_data()

    # 展示数据集
    with st.expander("查看数据集示例"):
        st.dataframe(data.head())
        st.write(f"数据集大小: {data.shape[0]}行 × {data.shape[1]}列")

    # 分析重要特征
    st.subheader("数据特征说明")
    st.markdown("""
    - **MedInc**: 区域内家庭收入中位数
    - **HouseAge**: 区域内房屋年龄中位数
    - **AveRooms**: 平均房间数
    - **AveBedrms**: 平均卧室数
    - **Population**: 区域人口
    - **AveOccup**: 平均入住率
    - **Latitude**: 纬度
    - **Longitude**: 经度
    """)

    # 训练模型
    with st.spinner("模型训练中..."):
        result = train_model(data, epochs=epochs, learning_rate=learning_rate, force_train=force_train)
        model = result['model']
        metrics = result['metrics']
        feature_importance = result['feature_importance']
        scaler = result['scaler']

    # 显示模型评估指标
    st.subheader("模型性能")
    metrics_df = pd.DataFrame({
        "指标": list(metrics.keys()),
        "值": list(metrics.values())
    })
    st.dataframe(metrics_df)

    # 显示特征重要性
    st.subheader("特征重要性")
    importance_df = pd.DataFrame({
        '特征': list(feature_importance.keys()),
        '重要性': list(feature_importance.values())
    }).sort_values('重要性', ascending=False)

    # 避免中文乱码，使用英文名称
    feature_name_map = {
        'MedInc': '收入中位数',
        'HouseAge': '房屋年龄',
        'AveRooms': '平均房间数',
        'AveBedrms': '平均卧室数',
        'Population': '区域人口',
        'AveOccup': '平均入住率',
        'Latitude': '纬度',
        'Longitude': '经度'
    }

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 只用英文特征名绘图
    top_features = importance_df['特征'][:5]
    ax.bar(top_features, importance_df['重要性'][:5])
    plt.xticks(rotation=45, ha='right')

    # 保存图表并显示
    st.pyplot(fig)

    # 添加中文注释
    feature_importance_text = ""
    for i, feature in enumerate(top_features, 1):
        feature_cn = feature_name_map.get(feature, feature)
        importance_val = importance_df[importance_df['特征'] == feature]['重要性'].values[0]
        feature_importance_text += f"{i}. {feature_cn} ({feature}): {importance_val:.4f}\n"

    st.text("前5个重要特征及其权重：")
    st.text(feature_importance_text)

    # 房价预测
    st.subheader("房价预测")
    st.write("请输入房屋特征进行预测：")

    # 创建输入框
    col1, col2 = st.columns(2)

    with col1:
        medinc = st.number_input("收入中位数 (MedInc)", value=3.5, min_value=0.0, max_value=15.0, step=0.1)
        house_age = st.number_input("房屋年龄 (HouseAge)", value=25.0, min_value=1.0, max_value=60.0, step=1.0)
        ave_rooms = st.number_input("平均房间数 (AveRooms)", value=5.0, min_value=1.0, max_value=10.0, step=0.1)
        ave_bedrms = st.number_input("平均卧室数 (AveBedrms)", value=1.0, min_value=0.1, max_value=5.0, step=0.1)

    with col2:
        population = st.number_input("区域人口 (Population)", value=1500.0, min_value=100.0, max_value=10000.0,
                                     step=100.0)
        ave_occup = st.number_input("平均入住率 (AveOccup)", value=3.0, min_value=0.5, max_value=10.0, step=0.1)
        latitude = st.number_input("纬度 (Latitude)", value=35.0, min_value=30.0, max_value=45.0, step=0.1)
        longitude = st.number_input("经度 (Longitude)", value=-120.0, min_value=-130.0, max_value=-110.0, step=0.1)

    # 预测按钮
    if st.button("预测房价", key="predict_button", type="primary"):
        # 构建特征数组
        features = [
            medinc, house_age, ave_rooms, ave_bedrms,
            population, ave_occup, latitude, longitude
        ]
        # 检查输入数据是否存在异常值
        if np.isnan(features).any() or np.isinf(features).any():
            st.error("输入数据包含异常值，请检查并重新输入。")
        else:
            # 预测
            prediction = predict_housing_price(model, features, scaler)
            # 显示结果
            st.success(f"预测房价: ${prediction * 100000:.2f}")

            # 可视化单一特征与房价的关系 - 使用英文标签避免乱码问题
            st.subheader("收入与房价关系")

            # 展示MedInc与房价的关系
            fig, ax = plt.subplots(figsize=(10, 6))
            income_values = np.linspace(0, 15, 100)
            prices = []

            for inc in income_values:
                tmp_features = features.copy()
                tmp_features[0] = inc
                price = predict_housing_price(model, tmp_features, scaler)
                st.write(f"收入值: {inc}, 预测房价: {price}")
                prices.append(price)

            ax.plot(income_values, prices)
            ax.axvline(x=medinc, color='red', linestyle='--')
            ax.set_xlabel('Income (MedInc)')  # 使用英文
            ax.set_ylabel('Price')  # 使用英文
            plt.tight_layout()
            st.pyplot(fig)

            # 添加中文说明
            st.write("""
            上图显示了收入中位数(MedInc)与房价的关系。红色虚线表示您输入的收入值。
            从图中可以看出，随着收入的增加，预测的房价也呈现上升趋势。
            """)

    st.markdown("<div class='footer'>© 2025 机器学习算法平台 | wangxianfu.top</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    render_california_housing_page()