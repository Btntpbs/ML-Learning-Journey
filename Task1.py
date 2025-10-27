
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 42
CSV_PATH = Path(r"C:\Users\pc\Desktop\classification_set.csv")

def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    label_candidates = [c for c in df.columns if c.lower() in ["label", "target", "class", "y"]]
    if label_candidates:
        y = df[label_candidates[0]]
        X = df.drop(columns=[label_candidates[0]])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    return X, y

def build_preprocessors(X):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")
    ohe = OneHotEncoder()

    preprocess_tree_rf = ColumnTransformer(
        transformers=[
            ("num", num_imputer, numeric_cols),
            ("cat", Pipeline([("impute", cat_imputer), ("ohe", ohe)]), categorical_cols),
        ],
        remainder="drop",
    )

    preprocess_sgd = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", num_imputer), ("scale", StandardScaler())]), numeric_cols),
            ("cat", Pipeline([("impute", cat_imputer), ("ohe", ohe)]), categorical_cols),
        ],
        remainder="drop",
    )

    return preprocess_tree_rf, preprocess_sgd

def print_confusion_matrix(y_true, y_pred, model_name):
    labels = sorted(pd.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Real_{l}" for l in labels],
        columns=[f"Pred_{l}" for l in labels]
    )
    print(f"\n--- {model_name} Confusion Matrix ---")
    print(cm_df)

def main():
    X, y = load_data(CSV_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    preprocess_tree_rf, preprocess_sgd = build_preprocessors(X)

    models = {
        "Decision Tree": Pipeline([
            ("prep", preprocess_tree_rf),
            ("clf", tree.DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]),
        "SGDClassifier": Pipeline([
            ("prep", preprocess_sgd),
            ("clf", SGDClassifier(random_state=RANDOM_STATE,
                                  max_iter=2000, tol=1e-3, loss="log_loss")),
        ]),
        "Random Forest": Pipeline([
            ("prep", preprocess_tree_rf),
            ("clf", RandomForestClassifier(random_state=RANDOM_STATE,
                                          n_estimators=300, n_jobs=-1)),
        ]),
    }

    best_model = None
    best_name = None
    best_acc = -1

    print("\n========== AI LAB CLASSIFICATION ==========")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))

        print_confusion_matrix(y_test, y_pred, name)

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name


    print("\n=== SUMMARY ===")
    print(f"Best Model: {best_name}")
    print(f"Best Accuracy: {best_acc:.4f}")

    print("\n=============================================================\n")

if __name__ == "__main__":
    main()

