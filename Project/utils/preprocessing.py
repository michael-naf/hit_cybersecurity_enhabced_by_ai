from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# -----------------------------
# Encode labels
# -----------------------------
def encode_labels(y, label_map={'benign':0, 'suspicious':1}):
    """
    Encode string labels to numeric
    """
    return y.map(label_map)

# -----------------------------
# Timestamp processing (fully cyclic)
# -----------------------------
def process_timestamp(X, timestamp_col):
    """
    Convert timestamp to datetime and extract numeric + cyclic features.
    Fully robust to ISO format (YYYY-MM-DD HH:MM:SS) and other formats.
    """
    X = X.copy()
    
    # -----------------------------
    # Convert to datetime
    # -----------------------------
    X[timestamp_col] = pd.to_datetime(X[timestamp_col], dayfirst=True, errors='coerce')

    # -----------------------------
    # Basic numeric features
    # -----------------------------
    X['year'] = X[timestamp_col].dt.year
    X['month'] = X[timestamp_col].dt.month
    X['day'] = X[timestamp_col].dt.day
    X['hour'] = X[timestamp_col].dt.hour
    X['minute'] = X[timestamp_col].dt.minute
    X['dayofweek'] = X[timestamp_col].dt.dayofweek  # Monday=0, Sunday=6
    # Shift to Sunday=0
    X['dayofweek'] = (X['dayofweek'] + 1) % 7
    X['is_weekend'] = X['dayofweek'].isin([0,6]).astype(int)

    # -----------------------------
    # Cyclic encoding
    # -----------------------------
    # Hour
    X['hour_sin'] = np.sin(2 * np.pi * X['hour']/24)
    X['hour_cos'] = np.cos(2 * np.pi * X['hour']/24)
    # Minute
    X['minute_sin'] = np.sin(2 * np.pi * X['minute']/60)
    X['minute_cos'] = np.cos(2 * np.pi * X['minute']/60)
    # Day of month (manual month lengths)
    month_days = {1:31,2:28,3:31,4:30,5:31,6:30,
                  7:31,8:31,9:30,10:31,11:30,12:31}
    days_in_month = X['month'].map(month_days)
    X['day_sin'] = np.sin(2 * np.pi * X['day'] / days_in_month)
    X['day_cos'] = np.cos(2 * np.pi * X['day'] / days_in_month)
    # Day of week
    X['dow_sin'] = np.sin(2 * np.pi * X['dayofweek']/7)
    X['dow_cos'] = np.cos(2 * np.pi * X['dayofweek']/7)
    # Month
    X['month_sin'] = np.sin(2 * np.pi * X['month']/12)
    X['month_cos'] = np.cos(2 * np.pi * X['month']/12)

    # -----------------------------
    # Drop raw columns
    # -----------------------------
    drop_cols = [timestamp_col, 'hour','minute','day','dayofweek','month']
    X = X.drop(columns=drop_cols, errors='ignore')  # safer in case some columns missing

    return X


# -----------------------------
# Handle missing values
# -----------------------------
def handle_missing_values(X, numeric_cols, categorical_cols):
    """
    Fill missing values in numeric and categorical columns
    """
    X = X.copy()
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    X[categorical_cols] = X[categorical_cols].fillna('Unknown')
    return X

# -----------------------------
# Handle outliers
# -----------------------------
def handle_outliers(X, numeric_cols):
    """
    Clip numeric features to remove extreme outliers (IQR method)
    """
    X = X.copy()
    Q1 = X[numeric_cols].quantile(0.25)
    Q3 = X[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    X[numeric_cols] = X[numeric_cols].clip(lower=lower, upper=upper, axis=1)
    return X

# -----------------------------
# Remove highly correlated features
# -----------------------------
def remove_highly_correlated(X, numeric_cols, threshold=0.9):
    """
    Remove highly correlated numeric features
    Returns: reduced X, updated numeric_cols, list of removed features
    """
    X = X.copy()
    corr_matrix = X[numeric_cols].corr().abs()
    upper_tri = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape)).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    X = X.drop(columns=to_drop)
    numeric_cols = [col for col in numeric_cols if col not in to_drop]
    return X, numeric_cols, to_drop

# -----------------------------
# Embed command_text semantically
# -----------------------------
def embed_command_text(X, text_col='command_text', model_name='all-MiniLM-L6-v2'):
    """
    Convert command_text into semantic embeddings using a pretrained sentence transformer
    Returns a DataFrame of embeddings
    """
    X = X.copy()
    model = SentenceTransformer(model_name, device="cpu")
    embeddings = model.encode(X[text_col].tolist(), show_progress_bar=True)
    X_emb = pd.DataFrame(embeddings, columns=[f"{text_col}_embed_{i}" for i in range(embeddings.shape[1])])
    X_emb.index = X.index
    return X_emb

# -----------------------------
# Full preprocessing pipeline
# -----------------------------
def preprocess_data(
    X,
    timestamp_col=None,
    text_col=None,
    label_col=None,
    numeric_cols=None,
    categorical_cols=None
):
    X = X.copy()

    # --------------------------
    # Auto-detect numeric/categorical if not provided
    # --------------------------
    if numeric_cols is None:
        numeric_cols = X.select_dtypes(include="number").columns.tolist()
    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include="object").columns.tolist()
        # remove label/text/timestamp if present
        categorical_cols = [col for col in categorical_cols if col not in [label_col, text_col, timestamp_col]]

    # --------------------------
    # Process timestamp
    # --------------------------
    if timestamp_col and timestamp_col in X.columns:
        X = process_timestamp(X, timestamp_col)
        # Add new time features to numeric_cols
        new_time_features = [
            'hour_sin','hour_cos',
            'minute_sin','minute_cos',
            'day_sin','day_cos',
            'dow_sin','dow_cos',
            'month_sin','month_cos'
        ]
        numeric_cols += new_time_features

    # --------------------------
    # Handle missing values
    # --------------------------
    X = handle_missing_values(X, numeric_cols, categorical_cols)

    # --------------------------
    # Handle outliers
    # --------------------------
    X = handle_outliers(X, numeric_cols)

    # --------------------------
    # Remove highly correlated features
    # --------------------------
    X, numeric_cols, dropped = remove_highly_correlated(X, numeric_cols)

    # --------------------------
    # Encode labels
    # --------------------------
    y_encoded = None
    if label_col and label_col in X.columns:
        y_encoded = encode_labels(X[label_col])
        X = X.drop(columns=[label_col])

    # --------------------------
    # Embed command_text
    # --------------------------
    X_text_emb = None
    if text_col and text_col in X.columns:
        X_text_emb = embed_command_text(X, text_col=text_col)

    return X, numeric_cols, y_encoded, X_text_emb
