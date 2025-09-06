# type: ignore
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os


class KNNClassifier:
    def __init__(self, n_neighbors=3):
        """
        初始化KNN分类器
        
        Args:
            n_neighbors (int): K值，默认为3
        """
        self.n_neighbors = n_neighbors
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.examples = {'A': [], 'B': [], 'C': []}
        self.is_trained = False
        
    def add_example(self, features, label):
        """
        添加训练样本
        
        Args:
            features (numpy.ndarray): 特征向量
            label (str): 标签 ('A', 'B', 或 'C')
            
        Returns:
            int: 该类别的样本总数
        """
        if label not in self.examples:
            raise ValueError(f"标签 {label} 无效，必须是 'A', 'B', 或 'C'")
        
        try:
            # 确保特征向量是一维的
            features_flat = features.flatten()
            
            print(f"添加{label}类样本: 特征维度={features_flat.shape}, 特征范围=[{features_flat.min():.4f}, {features_flat.max():.4f}]")
            
            # 将特征向量添加到对应类别
            self.examples[label].append(features_flat)
            
            print(f"{label}类样本总数: {len(self.examples[label])}")
            
            # 重新训练模型
            self._train_model()
            
            return len(self.examples[label])
            
        except Exception as e:
            print(f"添加样本时出错: {str(e)}")
            return len(self.examples[label])
    
    def _train_model(self):
        """训练KNN模型"""
        try:
            # 收集所有样本和标签
            all_features = []
            all_labels = []
            
            for label, features_list in self.examples.items():
                if features_list:  # 只处理有样本的类别
                    all_features.extend(features_list)
                    all_labels.extend([label] * len(features_list))
            
            print(f"训练数据: 总样本数={len(all_features)}, 类别分布={dict(zip(*np.unique(all_labels, return_counts=True)))}")
            
            if len(all_features) >= self.n_neighbors:
                # 转换为numpy数组
                X = np.array(all_features)
                y = np.array(all_labels)
                
                print(f"特征矩阵形状: {X.shape}")
                print(f"特征范围: [{X.min():.4f}, {X.max():.4f}]")
                
                # 标准化特征
                X_scaled = self.scaler.fit_transform(X)
                
                print(f"标准化后特征范围: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")
                
                # 训练模型
                self.classifier.fit(X_scaled, y)
                self.is_trained = True
                print("模型训练成功!")
            else:
                self.is_trained = False
                print(f"样本数量不足，需要至少 {self.n_neighbors} 个样本，当前只有 {len(all_features)} 个")
                
        except Exception as e:
            print(f"训练模型时出错: {str(e)}")
            self.is_trained = False
    
    def predict(self, features):
        """
        预测样本类别
        
        Args:
            features (numpy.ndarray): 特征向量
            
        Returns:
            tuple: (预测类别, 置信度)
        """
        if not self.is_trained:
            return None, 0.0
            
        try:
            # 确保特征向量是一维的
            features_flat = features.flatten()
            print(f"预测特征维度: {features_flat.shape}, 范围: [{features_flat.min():.4f}, {features_flat.max():.4f}]")
            
            # 标准化特征
            features_scaled = self.scaler.transform([features_flat])
            # 类型忽略：features_scaled 是 numpy 数组
            print(f"标准化后特征范围: [{float(features_scaled.min()):.4f}, {float(features_scaled.max()):.4f}]")  # type: ignore
            
            # 预测
            prediction = self.classifier.predict(features_scaled)[0]
            print(f"原始预测结果: {prediction}")
            
            # 获取预测概率（如果可用）
            if hasattr(self.classifier, 'predict_proba'):
                try:
                    probabilities = self.classifier.predict_proba(features_scaled)[0]
                    print(f"预测概率: {dict(zip(self.classifier.classes_, probabilities))}")
                except:
                    pass
            
            # 计算置信度（基于最近邻的距离）
            distances, indices = self.classifier.kneighbors(features_scaled)
            print(f"最近邻距离: {distances[0]}")
            print(f"最近邻索引: {indices[0]}")
            
            # 改进的置信度计算
            min_distance = np.min(distances[0])
            avg_distance = np.mean(distances[0])
            
            # 获取最近邻的标签
            neighbor_indices = indices[0]
            if hasattr(self.classifier, '_y') and self.classifier._y is not None:
                neighbor_labels = [self.classifier._y[idx] for idx in neighbor_indices]
            else:
                # 如果_y不可用，使用训练数据重建
                all_labels = []
                for label, features_list in self.examples.items():
                    if features_list:
                        all_labels.extend([label] * len(features_list))
                neighbor_labels = [all_labels[idx] for idx in neighbor_indices]
            print(f"最近邻标签: {neighbor_labels}")
            
            # 计算预测类别在最近邻中的比例
            predicted_count = sum(1 for label in neighbor_labels if label == prediction)
            neighbor_ratio = predicted_count / len(neighbor_labels)
            print(f"邻居一致性: {neighbor_ratio} ({predicted_count}/{len(neighbor_labels)})")
            
            # 基于距离和邻居一致性的置信度计算
            # 距离越小，置信度越高；邻居一致性越高，置信度越高
            distance_confidence = max(0.0, 1.0 - (avg_distance / 10.0))  # 调整距离阈值
            consistency_confidence = neighbor_ratio
            
            print(f"距离置信度: {distance_confidence:.4f}, 一致性置信度: {consistency_confidence:.4f}")
            
            # 综合置信度
            confidence = (distance_confidence * 0.3 + consistency_confidence * 0.7)
            
            # 确保置信度在合理范围内
            confidence = max(0.01, min(0.99, confidence))
            
            print(f"最终预测: {prediction}, 置信度: {confidence:.4f}")
            return prediction, confidence
            
        except Exception as e:
            print(f"预测时出错: {str(e)}")
            return None, 0.0
    
    def predict_class(self, features):
        """
        预测样本类别（与predict方法相同，为了兼容性）
        
        Args:
            features (numpy.ndarray): 特征向量
            
        Returns:
            tuple: (预测类别, 置信度)
        """
        return self.predict(features)
    
    def get_sample_counts(self):
        """
        获取各类别的样本数量
        
        Returns:
            dict: 各类别的样本数量
        """
        return {label: len(features) for label, features in self.examples.items()}
    
    def clear_samples(self):
        """清除所有训练样本"""
        self.examples = {'A': [], 'B': [], 'C': []}
        self.is_trained = False
    
    def save_model(self, filepath):
        """
        保存模型到文件
        
        Args:
            filepath (str): 保存路径
        """
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'examples': self.examples,
            'is_trained': self.is_trained,
            'n_neighbors': self.n_neighbors
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """
        从文件加载模型
        
        Args:
            filepath (str): 模型文件路径
        """
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.classifier = model_data['classifier']
            self.scaler = model_data['scaler']
            self.examples = model_data['examples']
            self.is_trained = model_data['is_trained']
            self.n_neighbors = model_data.get('n_neighbors', 3)
            return True
        return False 