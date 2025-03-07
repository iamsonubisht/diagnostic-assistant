from dependency_injector import containers, providers
from modules.data_processing import PatientDataProcessor
from modules.models import RandomForestTriageModel
from modules.allocation import HospitalResourceAllocator
from modules.forecasting import ResourceDemandForecaster
from modules.database import Database
from modules.caching import Cache

class Container(containers.DeclarativeContainer):
    """
    Dependency injection container for the Emergency System.
    Manages dependencies and configurations for the application.
    """

    # Load configuration from a YAML file
    config = providers.Configuration(yaml_files=["config/config.yaml"])

    # Data Processing Module
    data_processor = providers.Singleton(
        PatientDataProcessor,
        data_file=config.data.file
    )

    # Triage Prediction Module
    triage_model = providers.Singleton(
        RandomForestTriageModel,
        n_estimators=config.models.triage.n_estimators,
        random_state=config.models.triage.random_state
    )

    # Resource Allocation Module
    resource_allocator = providers.Factory(
        HospitalResourceAllocator,
        hospitals=config.hospitals,
        patients=config.patients
    )

    # Demand Forecasting Module
    demand_forecaster = providers.Singleton(
        ResourceDemandForecaster
    )

    # Database Module
    database = providers.Singleton(
        Database,
        db_url=config.database.url
    )

    # Caching Module
    cache = providers.Singleton(
        Cache,
        host=config.cache.host,
        port=config.cache.port,
        db=config.cache.db
    )
