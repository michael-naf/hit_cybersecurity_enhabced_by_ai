# main.py
import bootstrap

import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils import (
    preprocess_data,
    features_for_isolation_forest,
    features_for_svm,
    features_for_autoencoder,
)


from agents import (
    IsolationForestAgent,
    SVMAgent,
    AutoencoderAgent,
    MetaAgent,
)



def main():
    # --------------------------
    # 1. Load your dataset
    # --------------------------
    df = pd.read_csv("data/Enterprise_A.csv")

    # --------------------------
    # 2. Preprocess data
    # --------------------------
    # Preprocess numeric, categorical, timestamp, label, and embed text
    # Updated: unpack all 4 outputs
    X_structured, numeric_cols, y, X_text_emb = preprocess_data(
        df,
        timestamp_col="timestamp",
        label_col="is_anomaly",
        text_col="command_text",
        numeric_cols=None,       # auto-detect
        categorical_cols=None    # auto-detect
    )

    # --------------------------
    # 3. Standardize numeric features
    # --------------------------
    print("Numeric columns used:", numeric_cols)
    scaler = StandardScaler()
    X_structured[numeric_cols] = scaler.fit_transform(X_structured[numeric_cols])

    # --------------------------
    # 4. Prepare features for each agent
    # --------------------------
    X_if = features_for_isolation_forest(X_structured, numeric_cols)
    X_svm = features_for_svm(X_structured, numeric_cols)
    X_ae = features_for_autoencoder(X_structured, numeric_cols, X_text_emb=X_text_emb)

    # Build feature dict for meta-agent
    X_dict = {
        "IsolationForest": X_if,
        "OneClassSVM": X_svm,
        "Autoencoder": X_ae
    }

    # --------------------------
    # 5. Initialize agents
    # --------------------------
    if_agent = IsolationForestAgent(contamination=0.05)
    svm_agent = SVMAgent(nu=0.05)
    ae_agent = AutoencoderAgent(input_dim=X_ae.shape[1], epochs=50, latent_dim=16)

    agents = [if_agent, svm_agent, ae_agent]

    # --------------------------
    # 6. Initialize meta-agent
    # --------------------------
    meta_agent = MetaAgent(agents, weights=[0.3, 0.3, 0.4])

    # --------------------------
    # 7. Fit all agents
    # --------------------------
    meta_agent.fit(X_dict)  # pass y if needed by agents

    # --------------------------
    # 8. Compute final anomaly scores
    # --------------------------
    final_scores = meta_agent.score(X_dict)

    # --------------------------
    # 9. Optional: threshold for binary anomalies
    # --------------------------
    threshold = pd.Series(final_scores).quantile(0.95)  # top 5% = anomalies
    preds = (final_scores > threshold).astype(int)

    # --------------------------
    # 10. Save results
    # --------------------------
    results = df.copy()
    results["anomaly_score"] = final_scores
    results["predicted_anomaly"] = preds

    results.to_csv("anomaly_results.csv", index=False)
    print("Anomaly detection complete. Results saved to anomaly_results.csv")


if __name__ == "__main__":
    main()
