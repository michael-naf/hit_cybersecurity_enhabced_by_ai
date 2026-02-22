# base_agent.py
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for all model agents.
    Enforces a common interface across agents.
    """

    def __init__(self, name):
        self.name = name
        self.model = None

    @abstractmethod
    def fit(self, X):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def score(self, X):
        """
        Produce anomaly scores.
        Higher score = more anomalous.
        """
        pass

    def predict(self, X, threshold=None):
        """
        Optional binary prediction from anomaly scores.
        """
        scores = self.score(X)

        if threshold is None:
            return scores

        return (scores > threshold).astype(int)

    def get_name(self):
        """
        Return agent name.
        """
        return self.name
