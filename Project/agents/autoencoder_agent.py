# autoencoder_agent.py
import numpy as np
from tf_keras.models import Model
from tf_keras.layers import Input, Dense
from tf_keras.optimizers import Adam
from tf_keras.callbacks import EarlyStopping
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from .base_agent import BaseAgent


class AutoencoderAgent(BaseAgent):
    """
    Autoencoder-based anomaly detection agent.
    Anomaly score = reconstruction error.
    """

    def __init__(
        self,
        name="Autoencoder",
        input_dim=None,
        hidden_dims=(64, 32),
        latent_dim=16,
        learning_rate=1e-3,
        epochs=50,
        batch_size=32
    ):
        super().__init__(name)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.model = None

    def _build_model(self):
        """
        Build symmetric autoencoder.
        """
        inputs = Input(shape=(self.input_dim,))

        x = inputs
        for dim in self.hidden_dims:
            x = Dense(dim, activation="relu")(x)

        latent = Dense(self.latent_dim, activation="relu")(x)

        x = latent
        for dim in reversed(self.hidden_dims):
            x = Dense(dim, activation="relu")(x)

        outputs = Dense(self.input_dim, activation="linear")(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="mse"
        )

        return model

    def fit(self, X):
        """
        Train autoencoder on (mostly) normal data.
        """
        if self.input_dim is None:
            self.input_dim = X.shape[1]

        self.model = self._build_model()

        early_stop = EarlyStopping(
            monitor="loss",
            patience=5,
            restore_best_weights=True
        )

        self.model.fit(
            X,
            X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            callbacks=[early_stop],
            verbose=0
        )

    def score(self, X):
        """
        Return anomaly scores.
        Higher = more anomalous.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        reconstructions = self.model.predict(X, verbose=0)

        # Mean Squared Error per sample
        reconstruction_error = np.mean(
            np.square(X - reconstructions),
            axis=1
        )

        return reconstruction_error


    def predict(self, X, threshold):
        """
        Binary classification from anomaly scores.
        0 = normal
        1 = anomaly
        """
        scores = self.score(X)
        return (scores > threshold).astype(int)


    def evaluate(self, X, y_true, threshold):
        """
        Compute full evaluation metrics.
        
        Returns:
            dict containing:
                confusion_matrix
                accuracy
                precision
                recall
                f1
        """
        y_pred = self.predict(X, threshold)

        cm = confusion_matrix(y_true, y_pred)

        return {
            "confusion_matrix": cm,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }


    def compute_roc(self, X, y_true):
        """
        Compute ROC curve and AUC.
        
        Returns:
            fpr, tpr, thresholds, auc_score
        """
        scores = self.score(X)

        fpr, tpr, thresholds = roc_curve(y_true, scores)
        auc_score = auc(fpr, tpr)

        return fpr, tpr, thresholds, auc_score


    def find_best_threshold(self, X, y_true):
        """
        Automatically find best threshold based on F1 score.
        
        Returns:
            best_threshold, best_f1
        """
        scores = self.score(X)
        thresholds = np.unique(scores)

        best_f1 = 0
        best_threshold = thresholds[0]

        for t in thresholds:
            y_pred = (scores > t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t

        return best_threshold, best_f1