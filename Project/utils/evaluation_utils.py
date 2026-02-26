# utils/evaluation_utils.py
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def evaluate_agent(agent, X, y_true, threshold):
    """
    Evaluate a single agent.
    Prints confusion matrix, precision, recall, f1_score.
    """
    y_pred = agent.predict(X, threshold=threshold)
    
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"===== Evaluation: {agent.get_name()} =====")
    print("Confusion Matrix:")
    print(cm)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
    
    return {"cm": cm, "precision": precision, "recall": recall, "f1": f1}


def evaluate_ensemble(agents_list, X_dict, y_true, weights=None, contamination=0.05, threshold=None):
    """
    Evaluate an ensemble of agents.
    Uses percentile-based thresholding (default top contamination% = anomalies).
    Prints confusion matrix, precision, recall, f1_score for the ensemble.
    """
    num_agents = len(agents_list)
    
    if weights is None:
        weights = np.ones(num_agents) / num_agents
    else:
        weights = np.array(weights) / np.sum(weights)
    
    all_scores = []
    for agent in agents_list:
        X_agent = X_dict[agent.get_name()]
        scores = agent.score(X_agent)
        all_scores.append(scores)
    
    all_scores = np.array(all_scores)  # shape = (num_agents, num_samples)
    
    # Weighted sum across agents
    final_scores = np.dot(weights, all_scores)
    
    # Threshold based on contamination (percentile)
    if threshold is None:
        # אם יש 5% anomalies → threshold = 95th percentile
        threshold = np.percentile(final_scores, 100 * (1 - contamination))
    
    y_pred = (final_scores >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    ensemble_names = "+".join([agent.get_name() for agent in agents_list])
    print(f"===== Ensemble Evaluation: {ensemble_names} =====")
    print("Confusion Matrix:")
    print(cm)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
    
    return {"cm": cm, "precision": precision, "recall": recall, "f1": f1, "threshold": threshold}


def evaluate_all_combinations(agents, X_dict, y_true, contamination=0.05):
    """
    Evaluate all single agents and all combinations of 2 or 3 agents.
    Uses contamination-based thresholding automatically.
    """
    # Single agents
    for agent in agents:
        X_agent = X_dict[agent.get_name()]
        evaluate_agent(agent, X_agent, y_true, 
                       threshold=np.percentile(agent.score(X_agent), 100*(1-contamination)))
    
    # Combinations of 2 and 3
    for r in [2, 3]:
        for combo in itertools.combinations(agents, r):
            evaluate_ensemble(combo, X_dict, y_true, contamination=contamination)