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
        gamma="scale"
    ):
        super().__init__(name)

        self.model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma
        )

    def fit(self, X):
        """
        Train the One-Class SVM.
        X must be numeric features only.
        """
        self.model.fit(X)

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
