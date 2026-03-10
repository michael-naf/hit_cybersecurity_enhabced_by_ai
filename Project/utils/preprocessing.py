# preprocessing.py
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Encode labels
# -----------------------------
def encode_labels(y, label_map={'benign':0, 'suspicious':1}):
    return y.map(label_map)


# -----------------------------
# Timestamp processing (cyclic)
# -----------------------------
def process_timestamp(X, timestamp_col):
    X = X.copy()
    X[timestamp_col] = pd.to_datetime(X[timestamp_col], dayfirst=True, errors='coerce')
    X['year'] = X[timestamp_col].dt.year
    X['month'] = X[timestamp_col].dt.month
    X['day'] = X[timestamp_col].dt.day
    X['hour'] = X[timestamp_col].dt.hour
    X['minute'] = X[timestamp_col].dt.minute
    X['dow'] = (X[timestamp_col].dt.dayofweek + 1) % 7 #sunday=0 --> saturday=6
    X['is_weekend'] = X['dow'].isin([5,6]).astype(int)
    # Cyclic encoding
    X['hour_sin'] = np.sin(2*np.pi*X['hour']/24)
    X['hour_cos'] = np.cos(2*np.pi*X['hour']/24)
    X['minute_sin'] = np.sin(2*np.pi*X['minute']/60)
    X['minute_cos'] = np.cos(2*np.pi*X['minute']/60)
    month_days = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
    days_in_month = X['month'].map(month_days)
    X['day_sin'] = np.sin(2*np.pi*X['day']/days_in_month)
    X['day_cos'] = np.cos(2*np.pi*X['day']/days_in_month)
    X['dow_sin'] = np.sin(2*np.pi*X['dow']/7)
    X['dow_cos'] = np.cos(2*np.pi*X['dow']/7)
    X['month_sin'] = np.sin(2*np.pi*X['month']/12)
    X['month_cos'] = np.cos(2*np.pi*X['month']/12)
    drop_cols = [timestamp_col,'hour','minute','day','dow','month']
    X = X.drop(columns=drop_cols, errors='ignore')
    return X


# -----------------------------
# Handle missing values
# -----------------------------
def handle_missing_values(X, numeric_cols, categorical_cols):
    X = X.copy()
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    X[categorical_cols] = X[categorical_cols].fillna('Unknown')
    if 'command_text' in X.columns:
        X['command_text'] = X['command_text'].fillna('').astype(str)
    return X



# -----------------------------
# Remove highly correlated features
# -----------------------------
def remove_highly_correlated(X, numeric_cols, threshold=0.95):
    X = X.copy()
    corr_matrix = X[numeric_cols].corr().abs()
    upper_tri = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    X = X.drop(columns=to_drop)
    numeric_cols = [col for col in numeric_cols if col not in to_drop]
    return X, numeric_cols


# -----------------------------
# Clean text column
# -----------------------------
def clean_text_column(df, column):
    df[column] = df[column].astype(str).str.lower()
    df[column] = df[column].apply(lambda x: re.sub(r"[^\x20-\x7E]", " ", x))
    df[column] = df[column].apply(lambda x: re.sub(r"\s+", " ", x).strip())
    return df


# -----------------------------
# TF-IDF vectorization
# -----------------------------
def tfidf_vectorize(df, column='command_text', max_features=512):
    df = clean_text_column(df, column)
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(df[column])
    return pd.DataFrame(X_tfidf.toarray(), index=df.index,
                        columns=[f"{column}_tfidf_{i}" for i in range(X_tfidf.shape[1])])


# -----------------------------
# Full preprocessing + prepare dict for MetaAgent
# -----------------------------
def preprocess_for_metaagent(
    df,
    text_col='command_text',
    label_col=None,
    timestamp_col=None,
    tfidf_max_features=512
):
    """
    Returns: X_dict (dict of np.arrays), y_encoded (if label exists)
    Each agent receives numeric + TF-IDF
    """
    df = df.copy()

    # Encode labels
    y_encoded = None
    if label_col and label_col in df.columns:
        y_encoded = encode_labels(df[label_col])
        df = df.drop(columns=[label_col])

    # Identify numeric/categorical
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    if text_col in categorical_cols:
        categorical_cols.remove(text_col)

    # Process timestamp
    if timestamp_col and timestamp_col in df.columns:
        df = process_timestamp(df, timestamp_col)
        new_time_features = [
            'hour_sin','hour_cos','minute_sin','minute_cos',
            'day_sin','day_cos','dow_sin','dow_cos',
            'month_sin','month_cos'
        ]
        numeric_cols += new_time_features
        categorical_cols = [col for col in categorical_cols if col in df.columns]

    # Handle missing values
    df = handle_missing_values(df, numeric_cols, categorical_cols)

    # Frequency Encoding for categorical features
    if categorical_cols:
        for col in categorical_cols:
            freq_map = df[col].value_counts()
            df[col] = df[col].map(freq_map)

        numeric_cols += categorical_cols

    # Remove highly correlated numeric features
    df, numeric_cols = remove_highly_correlated(df, numeric_cols)

    # TF-IDF
    X_tfidf = tfidf_vectorize(df, column=text_col, max_features=tfidf_max_features) if text_col in df.columns else None

    # Numeric array
    X_numeric = df[numeric_cols].values if numeric_cols else None

    # ==========================
    # SCALING
    # ==========================
    if X_numeric is not None:
        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(X_numeric)

    # Combine numeric + TF-IDF
    if X_numeric is not None and X_tfidf is not None:
        X_full = np.hstack([X_numeric, X_tfidf.values])
    elif X_numeric is not None:
        X_full = X_numeric
    elif X_tfidf is not None:
        X_full = X_tfidf.values
    else:
        raise ValueError("No features available for preprocessing.")

    # Prepare X_dict for MetaAgent
    X_dict = {
        "IsolationForest": X_full,
        "Autoencoder": X_full,
        "OneClassSVM": X_full
    }

    return X_dict, y_encoded