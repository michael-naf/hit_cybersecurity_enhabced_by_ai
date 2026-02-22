# meta_agent.py
import numpy as np

from .base_agent import BaseAgent


class MetaAgent(BaseAgent):
    """
    Meta-agent that aggregates multiple anomaly detection agents.
    Uses ensemble scoring (mean, weighted, or custom) to produce
    a final anomaly score.
    """

    def __init__(self, agents, name="MetaAgent", weights=None):
        """
        :param agents: list of BaseAgent instances
        :param weights: optional list of weights for each agent
        """
        super().__init__(name)
        self.agents = agents

        if weights is None:
            # Equal weight by default
            self.weights = np.ones(len(agents)) / len(agents)
        else:
            assert len(weights) == len(agents), "weights length must match number of agents"
            self.weights = np.array(weights) / np.sum(weights)  # normalize to sum=1

    def fit(self, X_dict):
        """
        Fit all agents.
        :param X_dict: dict of {agent_name: X_features_for_agent}
        """
        for agent in self.agents:
            X_agent = X_dict[agent.get_name()]
            agent.fit(X_agent)

    def score(self, X_dict):
        """
        Return ensemble anomaly scores.
        :param X_dict: dict of {agent_name: X_features_for_agent}
        :return: np.array of final anomaly scores (higher = more anomalous)
        """
        all_scores = []

        for agent in self.agents:
            X_agent = X_dict[agent.get_name()]
            scores = agent.score(X_agent)
            all_scores.append(scores)

        all_scores = np.array(all_scores)  # shape: (num_agents, num_samples)

        # Weighted sum across agents
        final_scores = np.dot(self.weights, all_scores)

        return final_scores

    def predict(self, X_dict, threshold=None):
        """
        Optional binary prediction based on ensemble score.
        """
        final_scores = self.score(X_dict)

        if threshold is None:
            return final_scores

        return (final_scores > threshold).astype(int)
