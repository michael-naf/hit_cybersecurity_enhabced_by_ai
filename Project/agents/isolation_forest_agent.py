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

    def fit(self, X):
        """
        Train the Isolation Forest on numeric features only.
        """
        self.model.fit(X)

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
