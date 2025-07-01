import torch
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io
import cv2
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sklearn.model_selection import train_test_split

from app.ml_models import (
    KNNModel, KMeansModel, LogisticRegressionModel,
    LinearRegressionModel, RandomForestModel,
    load_iris_data, load_digits_data, load_california_housing_data
)
from joblib import load, dump
import os
from app.config import settings, SECRET_KEY, ALGORITHM

router = APIRouter(prefix="/ml", tags=["machine_learning"])

MODEL_DIR = settings.MODEL_SAVE_PATH
os.makedirs(MODEL_DIR, exist_ok=True)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# Token verification function
async def verify_token(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        return email
    except JWTError:
        raise credentials_exception

# KNN Router with token verification
@router.post("/knn/train")
async def knn_train(n_neighbors: int = 5, token: str = Depends(verify_token)):
    try:
        X, y, _ = load_iris_data()
        model = KNNModel(n_neighbors=n_neighbors)
        model.train(X, y)
        return {"message": "KNN model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knn/predict")
async def knn_predict(file: UploadFile = File(...), token: str = Depends(verify_token)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        X = df.values

        model_path = os.path.join(MODEL_DIR, 'knn_model.joblib')
        scaler_path = os.path.join(MODEL_DIR, 'knn_scaler.joblib')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise HTTPException(status_code=400, detail="Model not trained yet")

        model = load(model_path)
        scaler = load(scaler_path)

        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        return JSONResponse(content={"predictions": predictions.tolist()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# KMeans Router with token verification
@router.post("/kmeans/train")
async def kmeans_train(n_clusters: int = 3, token: str = Depends(verify_token)):
    try:
        X, _, _ = load_iris_data()
        model = KMeansModel(n_clusters=n_clusters)
        model.train(X)
        return {"message": "KMeans model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kmeans/predict")
async def kmeans_predict(file: UploadFile = File(...), token: str = Depends(verify_token)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        X = df.values

        model_path = os.path.join(MODEL_DIR, 'kmeans_model.joblib')
        scaler_path = os.path.join(MODEL_DIR, 'kmeans_scaler.joblib')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise HTTPException(status_code=400, detail="Model not trained yet")

        model = load(model_path)
        scaler = load(scaler_path)

        X_scaled = scaler.transform(X)
        clusters = model.predict(X_scaled)

        return JSONResponse(content={"clusters": clusters.tolist()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Logistic Regression Router with token verification
@router.post("/logistic_regression/train")
async def logistic_regression_train(epochs: int = 1000, lr: float = 0.01, token: str = Depends(verify_token)):
    try:
        X, y = load_digits_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegressionModel(input_dim=X.shape[1], output_dim=1)
        model.train_model(X_train, y_train, epochs=epochs, lr=lr)
        return {"message": "Logistic Regression model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logistic_regression/predict")
async def logistic_regression_predict(file: UploadFile = File(...), token: str = Depends(verify_token)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        X = df.values

        model_path = os.path.join(MODEL_DIR, 'logistic_regression_model.pt')
        scaler_path = os.path.join(MODEL_DIR, 'logistic_regression_scaler.joblib')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise HTTPException(status_code=400, detail="Model not trained yet")

        model = LogisticRegressionModel(input_dim=X.shape[1], output_dim=1)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        scaler = load(scaler_path)
        model.scaler = scaler

        predictions = model.predict(X)

        return JSONResponse(content={"predictions": predictions.flatten().tolist()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Linear Regression Router with token verification
@router.post("/linear_regression/train")
async def linear_regression_train(epochs: int = 1000, lr: float = 0.01, token: str = Depends(verify_token)):
    try:
        X, y = load_california_housing_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegressionModel(input_dim=X.shape[1])
        model.train_model(X_train, y_train, epochs=epochs, lr=lr)
        return {"message": "Linear Regression model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/linear_regression/predict")
async def linear_regression_predict(file: UploadFile = File(...), token: str = Depends(verify_token)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        X = df.values

        model_path = os.path.join(MODEL_DIR, 'linear_regression_model.pt')
        scaler_path = os.path.join(MODEL_DIR, 'linear_regression_scaler.joblib')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise HTTPException(status_code=400, detail="Model not trained yet")

        model = LinearRegressionModel(input_dim=X.shape[1])
        model.load_state_dict(torch.load(model_path))
        model.eval()

        scaler = load(scaler_path)
        model.scaler = scaler

        predictions = model.predict(X)

        return JSONResponse(content={"predictions": predictions.flatten().tolist()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Random Forest Router with token verification
@router.post("/random_forest/train")
async def random_forest_train(epochs: int = 100, lr: float = 0.01, token: str = Depends(verify_token)):
    try:
        X, y, class_names = load_iris_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestModel(input_dim=X.shape[1], output_dim=len(class_names))
        model.train_model(X_train, y_train, epochs=epochs, lr=lr)
        return {"message": "Random Forest model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/random_forest/predict")
async def random_forest_predict(file: UploadFile = File(...), token: str = Depends(verify_token)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        X = df.values

        model_path = os.path.join(MODEL_DIR, 'random_forest_model.pt')
        scaler_path = os.path.join(MODEL_DIR, 'random_forest_scaler.joblib')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise HTTPException(status_code=400, detail="Model not trained yet")

        _, _, class_names = load_iris_data()
        model = RandomForestModel(input_dim=X.shape[1], output_dim=len(class_names))
        model.load_state_dict(torch.load(model_path))
        model.eval()

        scaler = load(scaler_path)
        model.scaler = scaler

        predictions = model.predict(X)

        return JSONResponse(content={
            "predictions": predictions.tolist(),
            "class_names": class_names.tolist()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
