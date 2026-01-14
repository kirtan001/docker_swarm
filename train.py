import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

MODEL_PATH = "model.joblib"

def train():
    # Load built-in Iris dataset (stable & offline)
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    # Train-test split
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Iris model trained and saved at {MODEL_PATH}")

if __name__ == "__main__":
    train()
