# agents/meta_agent.py
import numpy as np
from .base_agent import BaseAgent

class MetaAgent(BaseAgent):
    """
    Meta-agent that aggregates multiple anomaly detection agents.

    Features:
    - Soft weighted sum (default)
    - Hard majority vote option (uses percentile-based threshold if no threshold provided)
    - Supports X_val for agents that need it (e.g., Autoencoder)
    - Can return raw scores or binary predictions automatically
    """

    def __init__(self, agents, name="MetaAgent", weights=None, voting="soft", contamination=0.05):
        """
        :param agents: list of BaseAgent instances
        :param weights: optional list of weights for each agent
        :param voting: 'soft' for weighted sum, 'hard' for majority vote
        :param contamination: fraction of data expected to be anomalies (for automatic threshold)
        """
        super().__init__(name)
        self.agents = agents
        self.voting = voting.lower()
        assert self.voting in ["soft", "hard"], "voting must be 'soft' or 'hard'"
        self.contamination = contamination

        if weights is None:
            self.weights = np.ones(len(agents)) / len(agents)
        else:
            assert len(weights) == len(agents), "weights length must match number of agents"
            self.weights = np.array(weights) / np.sum(weights)

    def fit(self, X_dict, X_val_dict=None):
        """
        Fit all agents.
        :param X_dict: dict of {agent_name: X_features_for_agent} for training
        :param X_val_dict: optional dict {agent_name: X_features_for_validation} for agents that support it
        """
        for agent in self.agents:
            agent_name = agent.get_name()
            if agent_name not in X_dict:
                raise ValueError(f"X_dict missing data for agent '{agent_name}'")

            if X_val_dict is not None and agent_name in X_val_dict:
                agent.fit(X_dict[agent_name], X_val=X_val_dict[agent_name])
            else:
                agent.fit(X_dict[agent_name])

    def score(self, X_dict):
        """
        Compute ensemble anomaly scores.
        :param X_dict: dict of {agent_name: X_features_for_agent}
        :return: np.array of final anomaly scores (higher = more anomalous)
        """
        all_scores = []
        for agent in self.agents:
            agent_name = agent.get_name()
            if agent_name not in X_dict:
                raise ValueError(f"X_dict missing data for agent '{agent_name}'")
            scores = agent.score(X_dict[agent_name])
            all_scores.append(scores)

        all_scores = np.array(all_scores)  # shape = (num_agents, num_samples)

        if self.voting == "soft":
            final_scores = np.dot(self.weights, all_scores)
        else:  # hard voting
            # compute binary predictions per agent using percentile threshold
            binary_preds = []
            for i, scores in enumerate(all_scores):
                threshold = np.percentile(scores, 100 * (1 - self.contamination))
                binary_preds.append((scores >= threshold).astype(int))
            binary_preds = np.array(binary_preds)
            final_scores = np.mean(binary_preds, axis=0)  # fraction of agents voting anomaly

        return final_scores

    def predict(self, X_dict, threshold=None):
        """
        Return binary predictions.
        - If threshold=None, computes automatic threshold using contamination,
        for both soft and hard voting.
        """
        final_scores = self.score(X_dict)

        if threshold is None:
            if self.voting == "hard":
                # majority vote
                return (final_scores > 0.5).astype(int)
            else:  # soft voting
                # compute automatic threshold from contamination
                auto_thresh = np.percentile(final_scores, 100 * (1 - self.contamination))
                return (final_scores >= auto_thresh).astype(int)

        # explicit threshold
        return (final_scores >= threshold).astype(int)