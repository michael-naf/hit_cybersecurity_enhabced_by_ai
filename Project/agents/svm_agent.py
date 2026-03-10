# svm_agent.py
import numpy as np
from sklearn.svm import OneClassSVM

from .base_agent import BaseAgent


class SVMAgent(BaseAgent):
    """
    One-Class SVM agent for anomaly detection.
    """

    def __init__(
        self,
        name="OneClassSVM",
        kernel="rbf",
        nu=0.05,
        contamination=0.05,
        gamma="auto"
    ):
        super().__init__(name)

        self.contamination=contamination
        self.model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma
        )


    # =========================
    # Training
    # =========================
    def fit(self, X):
        """
        Train the One-Class SVM.
        X must be numeric features only.
        """
        self.model.fit(X)


    # =========================
    # Scoring
    # =========================
    def score(self, X):
        """
        Return anomaly scores.
        Higher score = more anomalous.
        """

        # decision_function:
        #   positive → inlier
        #   negative → outlier
        scores = self.model.decision_function(X)

        # Flip sign so:
        #   higher = more anomalous
        return -scores


    # =========================
    # Prediction
    # =========================
    def predict(self, X, threshold=None):
        scores = self.score(X)
        
        if threshold is None:
            # אם לא נשלח threshold, השתמש ב-contamination default
            n_outliers = max(1, int(len(scores) * self.contamination))
            threshold = np.sort(scores)[-n_outliers]

        # עכשיו כן משתמשים ב-threshold שנשלח
        return (scores >= threshold).astype(int)