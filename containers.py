from de_injector import containers, providers
from modules.data_processing import PatientDataProcessor
from modules.models import RandomForestTriageModel
from modules.allocation import HospitalResourceAllocator
from modules.forecasting import ResourceDemandForecaster
from modules.database import Database
from modules.caching import Cache

class Container(containers.DeclarativeContainer):
    config = providers.Configuration(yaml_files=["config/config.yaml"])

    data_processor = providers.Singleton(
        PatientDataProcessor,
        data_file=config.data.file
    )

    triage_model = providers.Singleton(
        RandomForestTriageModel,
        n_estimators=config.models.triage.n_estimators,
        random_state=config.models.triage.random_state
    )

    resource_allocator = providers.Factory(
        HospitalResourceAllocator,
        hospitals=config.hospitals,
        patients=config.patients
    )

    demand_forecaster = providers.Singleton(ResourceDemandForecaster)

    database = providers.Singleton(
        Database,
        db_url=config.database.url
    )

    cache = providers.Singleton(
        Cache,
        host=config.cache.host,
        port=config.cache.port,
        db=config.cache.db
    )