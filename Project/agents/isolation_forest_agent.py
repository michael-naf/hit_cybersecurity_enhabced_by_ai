# isolation_forest_agent.py
import numpy as np
from sklearn.ensemble import IsolationForest

from .base_agent import BaseAgent


class IsolationForestAgent(BaseAgent):
    """
    Isolation Forest agent for anomaly detection.
    Higher score = more anomalous.
    """

    def __init__(
        self,
        name="IsolationForest",
        n_estimators=100,
        max_samples="auto",
        contamination=0.05,
        random_state=42
    ):
        super().__init__(name)

        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state
        )


    # =========================
    # Training
    # =========================
    def fit(self, X):
        """
        Train the Isolation Forest on numeric features only.
        """
        self.model.fit(X)


    # =========================
    # Scoring
    # =========================
    def score(self, X):
        """
        Return anomaly scores.
        Higher = more anomalous.
        """

        # sklearn: decision_function
        #   higher = more normal
        #   lower = more anomalous
        scores = self.model.decision_function(X)

        # Flip sign: now higher = more anomalous
        return -scores


    # =========================
    # Prediction
    # =========================
    def predict(self, X, threshold=None):
        scores = self.score(X)

        if threshold is None:
            # אם לא נשלח threshold, השתמש ב-contamination default
            n_outliers = max(1, int(len(scores) * self.model.contamination))
            threshold = np.sort(scores)[-n_outliers]

        # עכשיו כן משתמשים ב-threshold שנשלח
        return (scores >= threshold).astype(int)