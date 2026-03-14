# main.py
from bootstrap import setup_environment
setup_environment()

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score


from utils.preprocessing import preprocess_for_metaagent
from utils.evaluation_utils import evaluate_agent, evaluate_ensemble
import utils.report_generator as rg
import utils.best_hyperparams as bh

from agents import (
    IsolationForestAgent,
    SVMAgent,
    AutoencoderAgent,
    MetaAgent,
)


def main():
    # --------------------------
    # 0. Choose dataset
    # --------------------------
    dataset_file, dataset_letter = bh.select_dataset()
    results_folder = f"Results/Enterprise_{dataset_letter}"
    os.makedirs(results_folder, exist_ok=True)  
    # --------------------------
    # 1. Load dataset
    # --------------------------
    df = pd.read_csv(f"data/{dataset_file}")

    # --------------------------
    # 2. Preprocess data
    # --------------------------
    X_dict, y = preprocess_for_metaagent(
        df,
        text_col='command_text',
        label_col='is_anomaly',
        timestamp_col='timestamp',
        tfidf_max_features=512
    )
    y = pd.Series(y)


    # --------------------------
    # 3. Split dataset into train / val / test
    # --------------------------
    X_train_dict = {}
    X_val_dict = {}
    X_test_dict = {}
    y_train_dict = {}

    # Assuming all features are in X_dict
    for key, X in X_dict.items():

        # Autoencoder has special train set without anomalies
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        if key == "Autoencoder" or key == "OneClassSVM":
            benign_idx = y_train[y_train == 0].index
            X_train = X[benign_idx]
            y_train = y_train[benign_idx]

        X_train_dict[key] = X_train
        X_val_dict[key] = X_val
        X_test_dict[key] = X_test
        y_train_dict[key] = y_train

    # --------------------------
    # 4. Initialize agents
    # --------------------------

    # Find best nu for SVM
    X_svm = X_train_dict["OneClassSVM"]
    best_nu = bh.find_best_nu(SVMAgent, X_svm, X_val_dict["OneClassSVM"], y_val)

    # Find ideal n_estimators for IsolationForest
    X_if = X_train_dict["IsolationForest"]
    y_if = y_train_dict["IsolationForest"]
    best_n_estimators = bh.find_best_n_estimators_if(X_if, y_if)
    print(f"Using best_n_estimators={best_n_estimators} for IsolationForestAgent")

    if_agent = IsolationForestAgent(contamination=0.05, n_estimators=best_n_estimators)
    svm_agent = SVMAgent(nu=best_nu)
    ae_agent = AutoencoderAgent(
        input_dim=X_train_dict["Autoencoder"].shape[1],
        epochs=50,
        latent_dim=16
    )

    agents = [if_agent, svm_agent, ae_agent]

    # --------------------------
    # 5. Initialize MetaAgent
    # --------------------------
    meta_agent = MetaAgent(agents, weights=[0.33, 0.33, 0.34])

    # --------------------------
    # 6. Fit all agents
    # --------------------------
    meta_agent.fit(X_train_dict, X_val_dict=X_val_dict)

    # --------------------------
    # 7. Individual Evaluation
    # --------------------------
    print("\n=== Individual Agent Evaluation (Validation) ===\n")

    agents_info = []
    generated_png_files = []

    for agent in agents:
        agent_name = agent.get_name()

        # Validation set for threshold tuning
        X_val = X_val_dict[agent_name]

        # Get anomaly scores on validation
        scores_val = agent.score(X_val)

        # Find best threshold based on validation
        threshold, best_f1 = bh.find_best_threshold(scores_val, y_val)
        print(f"\n✅ {agent_name} Best Threshold = {threshold:.6f} with F1 = {best_f1:.4f}\n")

        # Test set evaluation using threshold found on validation
        X_test = X_test_dict[agent_name]

        preds = agent.predict(X_test, threshold=threshold)

        # Evaluate agent
        evaluate_agent(agent, X_test, y_test, threshold=threshold)

        # Metrics
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)

        # Generate confusion matrix graph
        graph_path = f"{results_folder}/{agent_name}_graph.png"
        rg.plot_confusion_matrix_graph(y_test, preds, agent_name, graph_path)
        generated_png_files.append(graph_path)

        agents_info.append({
            'name': agent_name,
            'metrics': {'Precision': precision, 'Recall': recall, 'F1_score': f1},
            'graph': graph_path
        })


    # --------------------------
    # 8. Find best weights + threshold for MetaAgent (validation only)
    # --------------------------
    best_weights, best_threshold, best_preds_val = bh.find_best_weights_and_threshold_for_meta_agent(
        meta_agent,
        X_val_dict,
        y_val
    )
    meta_agent.weights = best_weights

    print("\nConfusion Matrix (Best Ensemble on Validation):")
    print(confusion_matrix(y_val, best_preds_val))


    # --------------------------
    # 9. Final Ensemble Evaluation on test
    # --------------------------
    print("\n=== Final MetaAgent Evaluation (Test) ===\n")
    best_preds_test = evaluate_ensemble(
        agents,
        X_test_dict,
        y_test,
        weights=best_weights,
        threshold=best_threshold
    )

    ensemble_precision = best_preds_test["precision"]
    ensemble_recall = best_preds_test["recall"]
    ensemble_f1 = best_preds_test["f1"]
    ensemble_graph_path = f"{results_folder}/Ensemble_graph.png"
    rg.plot_confusion_matrix_graph(y_test, best_preds_test["prediction"], "Ensemble", ensemble_graph_path)
    generated_png_files.append(ensemble_graph_path)

    ensemble_info = {
        'metrics': {'Precision': ensemble_precision, 'Recall': ensemble_recall, 'F1_score': ensemble_f1},
        'graph': ensemble_graph_path
    }

    # --------------------------
    # 10. Save results
    # --------------------------
    final_scores = meta_agent.score(X_test_dict)
    test_idx = X_test_dict["Autoencoder"].index if isinstance(X_test_dict["Autoencoder"], pd.DataFrame) else np.arange(X_test_dict["Autoencoder"].shape[0])
    results = df.loc[test_idx].copy()
    results["anomaly_score"] = final_scores
    results["predicted_anomaly"] = best_preds_test["prediction"]

    output_file = f"{results_folder}/Enterprise_{dataset_letter}_anomaly_results.csv"
    results.to_csv(output_file, index=False)
    print(f"\n✅ Anomaly detection complete. Results saved to {output_file}")
        
    report_file = f"{results_folder}/Enterprise_{dataset_letter}_report.pdf"
    rg.generate_pdf_report(report_file, agents_info, ensemble_info, weights=best_weights)
    print(f"\n✅ PDF report generated: {report_file}")

    # Delete temporary PNG files
    for file_path in generated_png_files:
        if os.path.exists(file_path):
            os.remove(file_path)

    print("🗑️ Temporary PNG files deleted.")

    # --------------------------
    # 11. Export Metrics CSV
    # --------------------------
    metrics_csv_path = f"{results_folder}/Enterprise_{dataset_letter}_metrics_summary.csv"

    rg.export_metrics_to_csv(
        metrics_csv_path,
        agents_info,
        ensemble_info
    )

    print(f"✅ Metrics CSV exported: {metrics_csv_path}")



    # --------------------------
    # Compare enterprises
    # --------------------------
    rg.compare_enterprises("Results")

if __name__ == "__main__":
    main()