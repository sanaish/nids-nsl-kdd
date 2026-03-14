"""
src/preprocess.py
-----------------
Reusable preprocessing pipeline for the NSL-KDD NIDS project.

Usage (from a notebook or train.py):
    from src.preprocess import load_data, preprocess

    X_train, X_test, y_train, y_test = preprocess("data/raw/KDDTrain.csv",
                                                    "data/raw/KDDTest.csv")
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


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

# Categorical columns that need LabelEncoder
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# Full column list in CSV order (cols 1–43)
COL_NAMES = (
    BASIC_FEATURES +
    CONTENT_FEATURES +
    TIME_TRAFFIC_FEATURES +
    HOST_TRAFFIC_FEATURES +
    ["label", "difficulty"]
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw NSL-KDD CSV files and assign column names.
    Drops the difficulty score column (not a network feature).

    Parameters
    ----------
    train_path : path to KDDTrain.csv
    test_path  : path to KDDTest.csv

    Returns
    -------
    train_df, test_df — DataFrames with named columns, difficulty dropped
    """
    train = pd.read_csv(train_path, header=None, names=COL_NAMES)
    test  = pd.read_csv(test_path,  header=None, names=COL_NAMES)

    train.drop("difficulty", axis=1, inplace=True)
    test.drop("difficulty",  axis=1, inplace=True)

    print(f"[load_data] Train: {train.shape}  |  Test: {test.shape}")
    return train, test


def map_binary_labels(df: pd.DataFrame) -> pd.Series:
    """
    Collapse multi-class attack labels to binary.
        'normal' → 0
        any attack subtype → 1

    Parameters
    ----------
    df : DataFrame containing a 'label' column

    Returns
    -------
    pd.Series of int (0 or 1)
    """
    return df["label"].apply(lambda x: 0 if x == "normal" else 1)


def encode_categoricals(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Label-encode categorical columns.
    Encoder is fitted on training data ONLY to prevent data leakage.

    Parameters
    ----------
    X_train, X_test : feature DataFrames (label column already removed)

    Returns
    -------
    X_train_enc, X_test_enc : encoded DataFrames
    encoders                : dict of {col_name: fitted LabelEncoder}
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()
    encoders = {}

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(X_train[col])                  # fit on train only
        X_train[col] = le.transform(X_train[col])
        X_test[col]  = le.transform(X_test[col])
        encoders[col] = le
        print(f"[encode] {col:15s} → {len(le.classes_)} classes")

    return X_train, X_test, encoders


def normalise(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scale all features to [0, 1] using MinMaxScaler.
    Scaler is fitted on training data ONLY to prevent data leakage.

    Parameters
    ----------
    X_train, X_test : encoded feature DataFrames

    Returns
    -------
    X_train_scaled, X_test_scaled : normalised DataFrames
    scaler                        : fitted MinMaxScaler (reuse in inference)
    """
    scaler = MinMaxScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),    # fit + transform on train
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),         # transform only on test
        columns=X_test.columns
    )

    print(f"[normalise] Train range: [{X_train_scaled.min().min():.2f}, "
          f"{X_train_scaled.max().max():.2f}]")
    print(f"[normalise] Test  range: [{X_test_scaled.min().min():.4f}, "
          f"{X_test_scaled.max().max():.4f}]")

    return X_train_scaled, X_test_scaled, scaler


def check_duplicates(X_train: pd.DataFrame, X_test: pd.DataFrame) -> int:
    """
    Check for identical feature rows across the train/test boundary.
    Overlap would mean the model has seen test data, inflating metrics.

    Parameters
    ----------
    X_train, X_test : scaled feature DataFrames

    Returns
    -------
    int : number of test rows that are duplicates of train rows
    """
    train_tagged = X_train.copy(); train_tagged["__src__"] = "train"
    test_tagged  = X_test.copy();  test_tagged["__src__"]  = "test"

    combined = pd.concat([train_tagged, test_tagged], ignore_index=True)
    cols = X_train.columns.tolist()
    dups = combined[combined.duplicated(subset=cols, keep=False)]
    n_test_dups = len(dups[dups["__src__"] == "test"])

    if n_test_dups == 0:
        print("[duplicates] ✅ No train/test feature overlap detected.")
    else:
        print(f"[duplicates] ⚠️  {n_test_dups} test rows overlap with train.")

    return n_test_dups


def preprocess(
    train_path: str,
    test_path: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Full preprocessing pipeline — call this from notebooks or train.py.

    Steps:
        1. Load raw CSVs and assign column names
        2. Map labels to binary (0 = Normal, 1 = Attack)
        3. Separate features (X) and labels (y)
        4. Label-encode categorical features (fit on train only)
        5. MinMax-normalise all features (fit on train only)
        6. Check for train/test duplicate rows

    Parameters
    ----------
    train_path : path to KDDTrain.csv
    test_path  : path to KDDTest.csv

    Returns
    -------
    X_train, X_test : preprocessed feature DataFrames, shape (n, 41)
    y_train, y_test : binary label Series (0 / 1)
    """
    # Step 1 — Load
    train_df, test_df = load_data(train_path, test_path)

    # Step 2 — Binary labels
    y_train = map_binary_labels(train_df)
    y_test  = map_binary_labels(test_df)
    print(f"[labels] Train → Normal: {(y_train==0).sum():,}  "
          f"Attack: {(y_train==1).sum():,}")
    print(f"[labels] Test  → Normal: {(y_test==0).sum():,}  "
          f"Attack: {(y_test==1).sum():,}")

    # Step 3 — Separate X
    X_train = train_df.drop("label", axis=1)
    X_test  = test_df.drop("label",  axis=1)

    # Step 4 — Encode categoricals
    X_train, X_test, _ = encode_categoricals(X_train, X_test)

    # Step 5 — Normalise
    X_train, X_test, _ = normalise(X_train, X_test)

    # Step 6 — Duplicate check
    check_duplicates(X_train, X_test)

    print(f"\n[preprocess] Done. X_train: {X_train.shape}  "
          f"X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test