# feature_extraction.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Isolation Forest
# -----------------------------
def features_for_isolation_forest(X_structured, numeric_cols):
    """
    Isolation Forest:
    - Numeric features only
    - No scaling required
    """
    return X_structured[numeric_cols]


# -----------------------------
# SVM
# -----------------------------
def features_for_svm(X_structured, numeric_cols):
    """
    SVM:
    - Numeric features
    - Requires scaling
    """
    scaler = StandardScaler()
    X_num = X_structured[numeric_cols]
    X_scaled = scaler.fit_transform(X_num)
    return pd.DataFrame(X_scaled, columns=numeric_cols, index=X_structured.index)


# -----------------------------
# Autoencoder
# -----------------------------
def features_for_autoencoder(X_structured, numeric_cols, X_text_emb=None, use_text=True):
    """
    Autoencoder:
    - Numeric features
    - Optionally semantic embeddings
    """
    X_num = X_structured[numeric_cols]

    if use_text and X_text_emb is not None:
        return pd.concat([X_num, X_text_emb], axis=1)

    return X_num


# -----------------------------
# Meta-Agent Input
# -----------------------------
def features_for_meta_agent(model_outputs):
    """
    Meta-agent does NOT see raw data.
    It sees model-level outputs like:
    - anomaly scores
    - reconstruction errors
    - decision margins
    """
    return pd.DataFrame(model_outputs)
