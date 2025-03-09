import asyncio
from containers import Container
import logging

# Set up logging configuration to log to console
logging.basicConfig(
    level=logging.INFO,  # Set to INFO or DEBUG as needed
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the container
container = Container()
container.config.from_yaml("config/config.yaml")

async def main():
    try:
        # Load the data processor
        data_processor = container.data_processor()
        X, y = await data_processor.get_features_and_target()

        # Load and train the triage model
        triage_model = container.triage_model()
        triage_model.train(X, y)
        evaluation_report = triage_model.evaluate(X, y)
        logger.info("Triage Model Evaluation:")
        logger.info(evaluation_report)

        # Predict triage categories for all patients
        predicted_triage = triage_model.predict(X)

        # Resource allocation
        allocator = container.resource_allocator()
        allocations = await allocator.allocate_resources(predicted_triage)  # Pass predicted triage

        # Display all patient allocations
        print("Patient Allocations:")
        for patient_id, triage_category in enumerate(predicted_triage, start=1):
            if patient_id - 1 < len(allocations):  # Check if the index is within bounds
                allocated_hospital = allocations[patient_id - 1]  # Assuming allocations is a list
                print(f"Patient {patient_id} (Triage: {triage_category}) -> {allocated_hospital}")
            else:
                print(f"Patient {patient_id} (Triage: {triage_category}) -> No allocation available")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())