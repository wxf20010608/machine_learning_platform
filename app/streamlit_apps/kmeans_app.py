# type: ignore
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# 设置页面标题
st.set_page_config(page_title="K-Means聚类演示", layout="wide")

# 页面标题
st.title("K-Means聚类算法演示")

# 侧边栏 - 参数设置
st.sidebar.header("参数设置")

# 选择数据集
dataset_option = st.sidebar.selectbox(
    "选择数据集",
    ["生成随机数据", "上传CSV文件"]
)

if dataset_option == "生成随机数据":
    # 生成随机数据的参数
    n_samples = st.sidebar.slider("样本数量", 50, 1000, 300)
    n_clusters = st.sidebar.slider("真实聚类数", 2, 10, 4)
    cluster_std = st.sidebar.slider("聚类标准差", 0.5, 3.0, 1.0)
    random_state = st.sidebar.slider("随机种子", 0, 100, 42)
    
    # 生成随机数据
    X, y = make_blobs(n_samples=n_samples, centers=n_clusters, 
                      cluster_std=cluster_std, random_state=random_state)
    
    # 创建数据框
    df = pd.DataFrame(X, columns=['特征1', '特征2'])
    df['真实类别'] = y
    
    # 显示数据集信息
    st.subheader("数据集信息")
    st.write(f"样本数量: {n_samples}")
    st.write(f"特征数量: 2")
    st.write(f"真实聚类数: {n_clusters}")
    
    # 显示数据样本
    st.subheader("数据样本")
    st.dataframe(df.head())
    
else:
    # 上传CSV文件
    uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # 筛选数值类型的列
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        # 选择特征
        feature_cols = st.sidebar.multiselect(
            "选择特征列", numeric_cols.tolist(), 
            default=numeric_cols.tolist()[:2] if len(numeric_cols) >= 2 else numeric_cols.tolist()
        )
        
        if len(feature_cols) < 2:
            st.error("请至少选择两个数值类型的特征列")
            st.stop()
        
        # 显示数据集信息
        st.subheader("数据集信息")
        st.write(f"样本数量: {df.shape[0]}")
        st.write(f"可用数值特征数量: {len(numeric_cols)}")
        
        # 显示数据样本
        st.subheader("数据样本")
        st.dataframe(df[feature_cols].head())
        
        try:
            # 提取特征并确保是数值类型
            X = df[feature_cols].values.astype(float)
        except ValueError as e:
            st.error(f"数据转换错误：请确保选择的列只包含数值。\n错误详情：{str(e)}")
            st.stop()
    else:
        st.info("请上传CSV文件")
        st.stop()

# K-Means参数设置
st.sidebar.header("K-Means参数")
k_clusters = st.sidebar.slider("聚类数量 (K)", 2, 10, 4)
max_iter = st.sidebar.slider("最大迭代次数", 100, 1000, 300)
n_init = st.sidebar.slider("初始化运行次数", 1, 20, 10)
standardize = st.sidebar.checkbox("标准化数据", value=True)

# 运行K-Means
if st.sidebar.button("运行K-Means"):
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 数据预处理
    status_text.text("数据预处理中...")
    progress_bar.progress(10)
    
    if standardize and 'X' in locals():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # 训练模型
    status_text.text("训练K-Means模型中...")
    progress_bar.progress(30)
    
    kmeans = KMeans(n_clusters=k_clusters, max_iter=max_iter, 
                   n_init=int(n_init), random_state=42)
    kmeans.fit(X_scaled)
    
    progress_bar.progress(70)
    status_text.text("生成可视化结果...")
    
    # 获取聚类结果
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # 如果数据已标准化，将中心点转换回原始空间
    if standardize:
        centers = scaler.inverse_transform(centers)
    
    # 创建结果数据框
    if dataset_option == "生成随机数据":
        result_df = pd.DataFrame(X, columns=['特征1', '特征2'])
        result_df['预测类别'] = labels
        result_df['真实类别'] = y
    else:
        result_df = df.copy()
        result_df['预测类别'] = labels
    
    # 显示聚类结果
    st.subheader("聚类结果")
    st.dataframe(result_df.head(20))
    
    # 可视化聚类结果
    st.subheader("聚类可视化")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制数据点
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
    
    # 绘制聚类中心
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='聚类中心')
    
    # 添加图例
    legend1 = ax.legend(*scatter.legend_elements(), title="聚类")
    ax.add_artist(legend1)
    ax.legend()
    
    # 设置标题和标签
    ax.set_title('K-Means聚类结果')
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')
    
    # 显示图形
    st.pyplot(fig)
    
    # 计算聚类评估指标
    st.subheader("聚类评估")
    
    # 计算惯性（Inertia）- 样本到最近聚类中心的距离平方和
    inertia = kmeans.inertia_
    st.write(f"惯性 (Inertia): {inertia:.2f}")
    st.write("惯性越小表示聚类效果越好")
    
    # 如果有真实标签，计算调整兰德指数
    if dataset_option == "生成随机数据":
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(y, labels)
        st.write(f"调整兰德指数 (ARI): {ari:.4f}")
        st.write("ARI取值范围为[-1, 1]，值越接近1表示聚类结果与真实类别越一致")
    
    # 完成进度条
    progress_bar.progress(100)
    status_text.text("完成!")

# 添加K-Means算法说明
with st.expander("K-Means算法说明"):
    st.markdown("""
    ## K-Means聚类算法
    
    K-Means是一种常用的聚类算法，它将数据点分组为预定义数量(K)的聚类。算法通过迭代方式工作，目标是最小化每个数据点到其分配的聚类中心的距离平方和。
    
    ### 算法步骤:
    
    1. **初始化**: 随机选择K个数据点作为初始聚类中心
    2. **分配**: 将每个数据点分配到最近的聚类中心
    3. **更新**: 重新计算每个聚类的中心点（均值）
    4. **重复**: 重复步骤2和3，直到聚类分配不再改变或达到最大迭代次数
    
    ### 优点:
    
    - 简单易实现
    - 计算效率高
    - 对大数据集有良好的扩展性
    
    ### 局限性:
    
    - 需要预先指定聚类数量K
    - 对初始聚类中心的选择敏感
    - 倾向于发现球形聚类
    - 对异常值敏感
    
    ### 应用场景:
    
    - 客户细分
    - 图像压缩
    - 文档聚类
    - 异常检测
    """)

# 添加页脚
st.sidebar.markdown("---")
st.sidebar.info("机器学习平台 - K-Means聚类演示")
