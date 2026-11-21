"""SmartCare no-show pipeline: rebuilds the Task 03 preprocessing and the Task 05/06 Random Forest."""
import joblib
import numpy as np
import pandas as pd
import sklearn
from functools import partial
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

NUM = ["age", "waiting_days", "previous_appointments",
       "missed_previous_appointments", "previous_admissions"]
CAT = ["gender", "blood_group", "department", "diagnosis"]
# Best Random Forest parameters found by GridSearchCV in Task 05
RF_PARAMS = dict(n_estimators=200, max_depth=10, min_samples_split=2,
                 min_samples_leaf=1, random_state=42)


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else np.asarray(a)


def build_bundle(csv_path):
    df = pd.read_csv(csv_path).drop_duplicates()
    df = df.dropna(subset=["no_show"])
    df = df[df["no_show"].isin([0, 1])].copy()
    X, y = df[NUM + CAT], df["no_show"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler", RobustScaler())]), NUM),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    sel = SelectPercentile(partial(mutual_info_classif, random_state=42), percentile=80)

    A_tr = _dense(sel.fit_transform(pre.fit_transform(X_tr), y_tr))
    A_te = _dense(sel.transform(pre.transform(X_te)))
    names = pre.get_feature_names_out()[sel.get_support()]

    model = RandomForestClassifier(**RF_PARAMS).fit(A_tr, y_tr)
    pred, proba = model.predict(A_te), model.predict_proba(A_te)[:, 1]

    return {
        "preprocessor": pre, "selector": sel, "model": model,
        "feature_names": list(names),
        "X_train_selected": A_tr,
        "metrics": {"Accuracy": accuracy_score(y_te, pred),
                    "Precision": precision_score(y_te, pred),
                    "Recall": recall_score(y_te, pred),
                    "F1 Score": f1_score(y_te, pred),
                    "ROC-AUC": roc_auc_score(y_te, proba)},
        "options": {c: sorted(df[c].dropna().unique().tolist()) for c in CAT},
        "ranges": {c: (int(df[c].min()), int(df[c].max())) for c in NUM},
        "base_rate": float(y.mean()),
        "test_proba": proba,
        "sklearn_version": sklearn.__version__,
    }


def predict_patient(bundle, row: dict):
    X = pd.DataFrame([row])[NUM + CAT]
    A = _dense(bundle["selector"].transform(bundle["preprocessor"].transform(X)))
    return float(bundle["model"].predict_proba(A)[0, 1]), A