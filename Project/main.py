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
from utils.report_generator import plot_confusion_matrix_graph, generate_pdf_report
from utils.best_hyperparams import find_best_weights_and_threshold_for_meta_agent, select_dataset

from agents import (
    IsolationForestAgent,
    SVMAgent,
    AutoencoderAgent,
    MetaAgent,
)

# ===========================
# MAIN
# ===========================
def main():
    # --------------------------
    # 0. Choose dataset
    # --------------------------
    dataset_file, dataset_letter = select_dataset()
    results_folder = "Results"
    if (dataset_letter == 'A'):
        results_folder = "Results/Enterprise_A"
    elif (dataset_letter == 'B'):
        results_folder = "Results/Enterprise_B"
    elif (dataset_letter == 'C'):
        results_folder = "Results/Enterprise_C"        
    # --------------------------
    # 1. Load dataset
    # --------------------------
    df = pd.read_csv(f"data/{dataset_file}")

    # --------------------------
    # 2. Preprocess data
    # --------------------------
    X_dict, y = preprocess_for_metaagent(
        df,
        text_col='command_line',
        label_col='is_anomaly',
        timestamp_col='timestamp',
        tfidf_max_features=512
    )
    y = pd.Series(y)

    # --------------------------
    # 3. Prepare Autoencoder train set WITHOUT anomalies
    # --------------------------
    normal_idx = y[y == 0].index
    X_ae = X_dict["Autoencoder"]
    X_ae_normal = X_ae[normal_idx]

    X_ae_train, X_ae_val = train_test_split(
        X_ae_normal,
        test_size=0.2,
        random_state=42
    )

    X_dict_train = X_dict.copy()
    X_dict_train["Autoencoder"] = X_ae_train
    X_val_dict = {"Autoencoder": X_ae_val}

    # --------------------------
    # 4. Initialize agents
    # --------------------------
    if_agent = IsolationForestAgent(contamination=0.05)
    svm_agent = SVMAgent(nu=0.05)
    ae_agent = AutoencoderAgent(
        input_dim=X_dict["Autoencoder"].shape[1],
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
    meta_agent.fit(X_dict_train, X_val_dict=X_val_dict)

    # --------------------------
    # 7. Individual Evaluation
    # --------------------------
    print("\n=== Individual Agent Evaluation ===\n")

    agents_info = []
    generated_png_files = []
    for agent in agents:
        if agent.get_name() == "Autoencoder":
            X_eval = X_dict["Autoencoder"]
            y_eval = y
            scores_normal = agent.score(X_ae_val)
            threshold = scores_normal.mean() + 2*scores_normal.std()
        else:
            X_eval = X_dict[agent.get_name()]
            y_eval = y
            threshold = None

        print(f"{agent.get_name()} Threshold = {threshold}")
        evaluate_agent(agent, X_eval, y_eval, threshold=threshold)

        preds = agent.predict(X_eval, threshold=threshold) if threshold else agent.predict(X_eval, threshold=np.percentile(agent.score(X_eval), 95))
        precision = precision_score(y_eval, preds, zero_division=0)
        recall = recall_score(y_eval, preds, zero_division=0)
        f1 = f1_score(y_eval, preds, zero_division=0)

        graph_path = f"{results_folder}/{agent.get_name()}_graph.png"
        plot_confusion_matrix_graph(y_eval, preds, agent.get_name(), graph_path)

        generated_png_files.append(graph_path)

        agents_info.append({
            'name': agent.get_name(),
            'metrics': {'Precision': precision, 'Recall': recall, 'F1_score': f1},
            'graph': graph_path
        })

    # --------------------------
    # 8. Find best weights + threshold
    # --------------------------
    best_weights, best_threshold, best_preds = find_best_weights_and_threshold_for_meta_agent(
        meta_agent,
        X_dict,
        y
    )

    meta_agent.weights = best_weights

    print("\nConfusion Matrix (Best Ensemble):")
    print(confusion_matrix(y, best_preds))

    # --------------------------
    # 9. Final Ensemble Evaluation
    # --------------------------
    print("\n=== Final MetaAgent Evaluation ===\n")
    evaluate_ensemble(
        agents,
        X_dict,
        y,
        weights=best_weights,
        threshold=best_threshold
    )

    ensemble_precision = precision_score(y, best_preds, zero_division=0)
    ensemble_recall = recall_score(y, best_preds, zero_division=0)
    ensemble_f1 = f1_score(y, best_preds, zero_division=0)
    ensemble_graph_path = f"{results_folder}/Ensemble_graph.png"
    plot_confusion_matrix_graph(y, best_preds, "Ensemble", ensemble_graph_path)

    generated_png_files.append(ensemble_graph_path)
    
    ensemble_info = {
        'metrics': {'Precision': ensemble_precision, 'Recall': ensemble_recall, 'F1_score': ensemble_f1},
        'graph': ensemble_graph_path
    }

    # --------------------------
    # 10. Save results
    # --------------------------
    final_scores = meta_agent.score(X_dict)
    results = df.copy()
    results["anomaly_score"] = final_scores
    results["predicted_anomaly"] = best_preds

    output_file = f"{results_folder}/Enterprise_{dataset_letter}_anomaly_results.csv"
    results.to_csv(output_file, index=False)
    print(f"\n✅ Anomaly detection complete. Results saved to {output_file}")
    
    report_file = f"{results_folder}/Enterprise_{dataset_letter}_report.pdf"
    generate_pdf_report(report_file, agents_info, ensemble_info)
    print(f"\n✅ PDF report generated: {report_file}")

    # Delete temporary PNG files
    for file_path in generated_png_files:
        if os.path.exists(file_path):
            os.remove(file_path)

    print("🗑️ Temporary PNG files deleted.")    

if __name__ == "__main__":
    main()