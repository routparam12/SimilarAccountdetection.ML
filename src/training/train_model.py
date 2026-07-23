import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.feature.feature_builder import create_features
from src.utils.config import (
    FEATURE_COLUMNS_PATH,
    MODEL_PATH,
    TRAIN_DATA_PATH,
)


def train():
    df = pd.read_csv(TRAIN_DATA_PATH)

    print("Generating Features...")
    X = df.apply(create_features, axis=1)
    y = df["score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    print("Training Model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"MAE: {mae:.2f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(FEATURE_COLUMNS_PATH, "wb") as f:
        pickle.dump(list(X.columns), f)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
