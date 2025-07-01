import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_iris, load_digits, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from joblib import dump, load
import os
from app.config import settings

MODEL_DIR = settings.MODEL_SAVE_PATH
os.makedirs(MODEL_DIR, exist_ok=True)


# KNN Model (using scikit-learn as it's more efficient for KNN)
class KNNModel:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()

    def train(self, X, y):
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        dump(self.model, os.path.join(MODEL_DIR, 'knn_model.joblib'))
        dump(self.scaler, os.path.join(MODEL_DIR, 'knn_scaler.joblib'))

    def predict(self, X):
        X = self.scaler.transform(X)
        model = load(os.path.join(MODEL_DIR, 'knn_model.joblib'))
        return model.predict(X)


# KMeans Model (using scikit-learn)
class KMeansModel:
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters)
        self.scaler = StandardScaler()

    def train(self, X):
        X = self.scaler.fit_transform(X)
        self.model.fit(X)
        dump(self.model, os.path.join(MODEL_DIR, 'kmeans_model.joblib'))
        dump(self.scaler, os.path.join(MODEL_DIR, 'kmeans_scaler.joblib'))

    def predict(self, X):
        X = self.scaler.transform(X)
        model = load(os.path.join(MODEL_DIR, 'kmeans_model.joblib'))
        return model.predict(X)


# PyTorch Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.scaler = StandardScaler()

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def train_model(self, X, y, epochs=1000, lr=0.01):
        X = self.scaler.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        criterion = nn.BCELoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        torch.save(self.state_dict(), os.path.join(MODEL_DIR, 'logistic_regression_model.pt'))
        dump(self.scaler, os.path.join(MODEL_DIR, 'logistic_regression_scaler.joblib'))

    def predict(self, X):
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self(X)
            predicted = (outputs >= 0.5).float()
        return predicted.numpy()


# PyTorch Linear Regression Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.scaler = StandardScaler()

    def forward(self, x):
        return self.linear(x)

    def train_model(self, X, y, epochs=1000, lr=0.01):
        X = self.scaler.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        torch.save(self.state_dict(), os.path.join(MODEL_DIR, 'linear_regression_model.pt'))
        dump(self.scaler, os.path.join(MODEL_DIR, 'linear_regression_scaler.joblib'))

    def predict(self, X):
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self(X)
        return outputs.numpy()


# PyTorch Random Forest-like Model (using a simple neural network as approximation)
class RandomForestModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RandomForestModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, output_dim)
        self.scaler = StandardScaler()

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return torch.softmax(self.output(x), dim=1)

    def train_model(self, X, y, epochs=100, lr=0.01):
        X = self.scaler.fit_transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        torch.save(self.state_dict(), os.path.join(MODEL_DIR, 'random_forest_model.pt'))
        dump(self.scaler, os.path.join(MODEL_DIR, 'random_forest_scaler.joblib'))

    def predict(self, X):
        X = self.scaler.transform(X)
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            outputs = self(X)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()


# Helper functions to load datasets
def load_iris_data():
    data = load_iris()
    X, y = data.data, data.target
    return X, y, data.target_names


def load_digits_data():
    data = load_digits()
    X, y = data.data, data.target
    return X, y


def load_california_housing_data():
    data = fetch_california_housing()
    X, y = data.data, data.target
    return X, y