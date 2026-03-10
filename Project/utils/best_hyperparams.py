#best_hyperparams.py
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.ensemble import IsolationForest


def find_best_n_estimators_if(X, y, n_start=50, n_end=300, step=10, contamination=0.05, random_state=42):
    """
    מחפש את מספר העצים האידיאלי ל-IsolationForest לפי F1.
    בודק מספרים מ-n_start עד n_end עם צעד step.
    """
    best_f1 = -1
    best_n = None

    print("\n=== Searching Best n_estimators for IsolationForest ===\n")

    n_options = np.arange(n_start, n_end + 1, step)
    
    for n in n_options:
        model = IsolationForest(n_estimators=n, contamination=contamination, random_state=random_state)
        model.fit(X)
        scores = -model.score_samples(X)  # higher score = more anomalous
        threshold = np.percentile(scores, 100 * (1 - contamination))
        preds = (scores >= threshold).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        print(f"n_estimators={n} -> F1={f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_n = n

    print(f"\n✅ Best n_estimators = {best_n} with F1 = {best_f1:.4f}\n")
    return best_n



def find_best_nu(svm_agent_class, X, y_true, nu_options=None):
    """
    מוצא את ערך ה-nu האידיאלי עבור SVM Agent לפי F1.
    מתאים ל-One-Class SVM (fit מקבל רק X).
    """

    if nu_options is None:
        nu_options = np.arange(0.01, 0.51, 0.01)  # 50 ערכים

    best_f1 = -1
    best_nu = None

    print("\n=== Searching Best Nu for SVM Agent ===\n")

    for nu in nu_options:
        agent = svm_agent_class(nu=nu)

        # 🔥 fit רק על X
        agent.fit(X)

        scores = agent.score(X)

        # threshold דיפולטי (כמו שאתה עושה כבר)
        _, f1 = find_best_threshold(scores, y_true, n_thresholds=50)

        print(f"nu={nu:.2f} -> F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_nu = nu

    print(f"\n✅ Best Nu = {best_nu:.2f} with F1 = {best_f1:.4f}\n")
    print(f"Using best nu={best_nu} for SVM (F1={best_f1:.4f})")


    return best_nu



def find_best_threshold(scores, y_true, n_thresholds=200):

    best_f1 = -1
    best_threshold = None

    percentiles = np.linspace(1, 99, n_thresholds)
    thresholds = np.percentile(scores, percentiles)

    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1

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