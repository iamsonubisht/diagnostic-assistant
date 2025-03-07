from prophet import Prophet
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ResourceDemandForecaster:
    """Forecasts resource demand using Prophet."""

    def __init__(self):
        self.model = Prophet()

    def train(self, dates, resource_demand):
        """Train the forecasting model."""
        df = pd.DataFrame({'ds': dates, 'y': resource_demand})
        self.model.fit(df)
        logger.info("Resource demand forecaster trained successfully.")

    def forecast(self, periods=7):
        """Generate a resource demand forecast."""
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        logger.info("Resource demand forecast generated.")
        return forecast