# autoencoder_agent.py
import numpy as np
import tensorflow as tf

from tf_keras.models import Model
from tf_keras.layers import Input, Dense
from tf_keras.optimizers import Adam
from tf_keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler

from .base_agent import BaseAgent


class AutoencoderAgent(BaseAgent):
    """
    Production-ready Autoencoder anomaly detector.
    - Includes scaling
    - Uses validation loss
    - Stable random seed
    """

    def __init__(
        self,
        name="Autoencoder",
        input_dim=None,
        hidden_dims=(64, 32),
        latent_dim=16,
        learning_rate=1e-3,
        epochs=100,
        batch_size=32,
        seed=42
    ):
        super().__init__(name)

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        self.model = None
        self.scaler = StandardScaler()

        # Reproducibility
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    # =========================
    # Model Building
    # =========================
    def _build_model(self):
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

    # =========================
    # Training
    # =========================
    def fit(self, X_train, X_val=None):

        if self.input_dim is None:
            self.input_dim = X_train.shape[1]

        # Fit scaler only on training data
        X_train = self.scaler.fit_transform(X_train)

        # apply same scaling to validation data
        if X_val is not None:
            X_val = self.scaler.transform(X_val)

        self.model = self._build_model()

        #if there is validation data, early stop according to val_loss, else early stop according to loss
        early_stop = EarlyStopping(
            monitor="val_loss" if X_val is not None else "loss",
            patience=10,
            restore_best_weights=True
        )

        self.model.fit(
            X_train,
            X_train,
            validation_data=(X_val, X_val) if X_val is not None else None,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            callbacks=[early_stop],
            verbose=0
        )

    # =========================
    # Scoring
    # =========================
    def score(self, X):

        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        reconstructions = self.model.predict(X_scaled, verbose=0)

        # calculating MSE for each sample
        reconstruction_error = np.mean(
            np.square(X_scaled - reconstructions),
            axis=1
        )

        return reconstruction_error

    # =========================
    # Prediction
    # =========================
    def predict(self, X, threshold=None):

        if threshold is None:
            raise ValueError(
                "Threshold not provided. Decide threshold in MetaAgent."
            )

        scores = self.score(X)
        return (scores > threshold).astype(int)