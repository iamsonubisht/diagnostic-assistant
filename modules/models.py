from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import logging

logger = logging.getLogger(__name__)

class RandomForestTriageModel:
    """Random Forest implementation of the triage model."""

    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def train(self, X_train, y_train):
        """Train the model on the given data."""
        self.model.fit(X_train, y_train)
        logger.info("Triage model trained successfully.")

    def predict(self, X_test):
        """Predict triage categories for the given data."""
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        """Evaluate the model's performance."""
        y_pred = self.predict(X_test)
        logger.info("Triage model evaluation completed.")
        return classification_report(y_test, y_pred)