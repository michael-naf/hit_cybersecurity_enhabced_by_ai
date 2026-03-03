#best_hyperparams.py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score


# =========================================================
# Search best weights + threshold
# =========================================================
def find_best_weights_and_threshold_for_meta_agent(meta_agent, X_dict, y):

    print("\n=== Searching Best Weights + Threshold ===\n")

    weight_options = np.arange(0.0, 1.1, 0.1)

    all_weight_combinations = [
        (w1, w2, w3)
        for w1 in weight_options
        for w2 in weight_options
        for w3 in weight_options
        if np.isclose(w1 + w2 + w3, 1.0)
    ]

    print(f"Total weight combinations to test: {len(all_weight_combinations)}")

    best_f1 = -1
    best_weights = None
    best_threshold = None
    best_preds = None

    for w1, w2, w3 in tqdm(all_weight_combinations, desc="Searching Weights"):

        meta_agent.weights = [w1, w2, w3]
        scores = meta_agent.score(X_dict)

        thresholds = np.linspace(scores.min(), scores.max(), 50)
        
        for t in thresholds:
            preds = (scores >= t).astype(int)
            f1 = f1_score(y, preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_weights = [w1, w2, w3]
                best_threshold = t
                best_preds = preds

    print("\n✅ Best Weights Found:", best_weights)
    print(f"✅ Best Threshold: {best_threshold:.6f}")
    print(f"✅ Best F1 Score: {best_f1:.6f}")

    return best_weights, best_threshold, best_preds


# =========================================================
# Select Enterprise
# =========================================================

def select_dataset():
    """ Allow user to choose which dataset to check """
    options = {"A": "Enterprise_A.csv",
               "B": "Enterprise_B.csv",
               "C": "Enterprise_C.csv"}
    
    while True:
        choice = input("Please choose dataset (A/B/C): ").strip().upper()
        if choice in options:
            filename = options[choice]
            print(f"\n✅ Chosen dataset: {filename}\n")
            return filename, choice
        else:
            print("❌ Available options: A / B / C.\nTry Again.")