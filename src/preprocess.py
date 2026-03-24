"""
src/preprocess.py
-----------------
Reusable preprocessing pipeline for the NSL-KDD NIDS project.

Encoding strategy (Fix #4):
    KNN and SVM use One-Hot Encoding (OHE) for categorical features.
    Naive Bayes keeps Label Encoding (LE) because OHE produces binary
    columns that violate the Gaussian distribution assumption, causing
    F1 to collapse from 0.77 to 0.38.

    Use preprocess()     for Naive Bayes  (returns 41 LE features).
    Use preprocess_ohe() for KNN and SVM  (returns 122 OHE features).

Usage:
    from src.preprocess import preprocess, preprocess_ohe

    X_train_le,  X_test_le,  y_train, y_test = preprocess("data/raw/KDDTrain.csv", "data/raw/KDDTest.csv")
    X_train_ohe, X_test_ohe, y_train, y_test = preprocess_ohe("data/raw/KDDTrain.csv", "data/raw/KDDTest.csv")
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder


# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

BASIC_FEATURES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
]
CONTENT_FEATURES = [
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
]
TIME_TRAFFIC_FEATURES = [
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate",
]
HOST_TRAFFIC_FEATURES = [
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

COL_NAMES = (
    BASIC_FEATURES + CONTENT_FEATURES +
    TIME_TRAFFIC_FEATURES + HOST_TRAFFIC_FEATURES +
    ["label", "difficulty"]
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_data(train_path: str, test_path: str) -> tuple:
    train = pd.read_csv(train_path, header=None, names=COL_NAMES)
    test  = pd.read_csv(test_path,  header=None, names=COL_NAMES)
    train.drop("difficulty", axis=1, inplace=True)
    test.drop("difficulty",  axis=1, inplace=True)
    print(f"[load_data] Train: {train.shape}  |  Test: {test.shape}")
    return train, test


def map_binary_labels(df: pd.DataFrame) -> pd.Series:
    return df["label"].apply(lambda x: 0 if x == "normal" else 1)


def normalise(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    scaler = MinMaxScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_te = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)
    print(f"[normalise] Train range: [{X_tr.min().min():.2f}, {X_tr.max().max():.2f}]")
    print(f"[normalise] Test  range: [{X_te.min().min():.4f}, {X_te.max().max():.4f}]")
    return X_tr, X_te, scaler


def check_duplicates(X_train: pd.DataFrame, X_test: pd.DataFrame) -> int:
    tr = X_train.copy(); tr["__src__"] = "train"
    te = X_test.copy();  te["__src__"] = "test"
    combined = pd.concat([tr, te], ignore_index=True)
    cols = X_train.columns.tolist()
    dups = combined[combined.duplicated(subset=cols, keep=False)]
    n = len(dups[dups["__src__"] == "test"])
    if n == 0:
        print("[duplicates] No train/test feature overlap detected.")
    else:
        print(f"[duplicates] WARNING: {n} test rows overlap with train.")
    return n


# ---------------------------------------------------------------------------
# Label Encoding  (Naive Bayes)
# ---------------------------------------------------------------------------

def encode_categoricals_le(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Label-encode categorical columns. For Naive Bayes only.
    Fitted on train only. Unseen test labels mapped to -1 (no crash).
    Returns 41-feature DataFrames + encoders dict.
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(X_train[col])
        X_train[col] = le.transform(X_train[col])
        known  = set(le.classes_)
        unseen = set(X_test[col].unique()) - known
        if unseen:
            print(f"[encode_le] {col}: WARNING {len(unseen)} unseen label(s) -> -1: {unseen}")
            X_test[col] = X_test[col].apply(
                lambda v: int(le.transform([v])[0]) if v in known else -1
            )
        else:
            X_test[col] = le.transform(X_test[col])
        encoders[col] = le
        print(f"[encode_le] {col:15s} -> {len(le.classes_)} classes")
    return X_train, X_test, encoders


# ---------------------------------------------------------------------------
# One-Hot Encoding  (KNN + SVM)
# ---------------------------------------------------------------------------

def encode_categoricals_ohe(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    One-Hot Encode categorical columns. For KNN and SVM.

    Eliminates false ordinal relationships imposed by LabelEncoder.
    protocol_type(3) + service(70) + flag(11) = 84 binary cols,
    expanding total features from 41 to 122.

    handle_unknown='ignore' zeros out any unseen category in test set —
    principled fallback, no data leakage.
    Fitted on training data ONLY.
    Returns 122-feature DataFrames + fitted OneHotEncoder.
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(X_train[CATEGORICAL_COLS])

    ohe_cols = ohe.get_feature_names_out(CATEGORICAL_COLS).tolist()

    def _apply(X):
        ohe_part = pd.DataFrame(
            ohe.transform(X[CATEGORICAL_COLS]),
            columns=ohe_cols, index=X.index
        )
        return pd.concat([X.drop(CATEGORICAL_COLS, axis=1), ohe_part], axis=1)

    X_train_enc = _apply(X_train)
    X_test_enc  = _apply(X_test)

    n_cats = sum(len(c) for c in ohe.categories_)
    print(f"[encode_ohe] {len(CATEGORICAL_COLS)} cat cols -> {n_cats} OHE cols. "
          f"Total features: {X_train_enc.shape[1]}")
    return X_train_enc, X_test_enc, ohe


# ---------------------------------------------------------------------------
# Public pipelines
# ---------------------------------------------------------------------------

def _base_pipeline(train_path, test_path, encode_fn, label):
    train_df, test_df = load_data(train_path, test_path)
    y_train = map_binary_labels(train_df)
    y_test  = map_binary_labels(test_df)
    print(f"[labels] Train -> Normal: {(y_train==0).sum():,}  Attack: {(y_train==1).sum():,}")
    print(f"[labels] Test  -> Normal: {(y_test==0).sum():,}  Attack: {(y_test==1).sum():,}")
    X_train = train_df.drop("label", axis=1)
    X_test  = test_df.drop("label",  axis=1)
    X_train, X_test, _ = encode_fn(X_train, X_test)
    X_train, X_test, _ = normalise(X_train, X_test)
    check_duplicates(X_train, X_test)
    print(f"\n[preprocess] Done ({label}). X_train: {X_train.shape}  X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def preprocess(train_path: str, test_path: str) -> tuple:
    """Label-Encoding pipeline — use for Naive Bayes. Returns 41 features."""
    return _base_pipeline(train_path, test_path, encode_categoricals_le, "LE")


def preprocess_ohe(train_path: str, test_path: str) -> tuple:
    """One-Hot Encoding pipeline — use for KNN and SVM. Returns 122 features."""
    return _base_pipeline(train_path, test_path, encode_categoricals_ohe, "OHE")
