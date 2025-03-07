from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from containers import Container
import asyncio
import logging

logger = logging.getLogger(__name__)
app = FastAPI()

container = Container()
container.config.from_yaml("config/config.yaml")

class PatientRequest(BaseModel):
    heart_rate: int
    blood_pressure: int
    oxygen_level: int
    injury_severity: int

@app.post("/triage")
async def predict_triage(patient: PatientRequest):
    try:
        model = container.triage_model()
        X = [[patient.heart_rate, patient.blood_pressure, patient.oxygen_level, patient.injury_severity]]
        prediction = await asyncio.to_thread(model.predict, X)
        return {"triage_category": prediction[0]}
    except Exception as e:
        logger.error(f"Error predicting triage: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/allocate-resources")
async def allocate_resources(patients: list[PatientRequest]):
    try:
        allocator = container.resource_allocator()
        patients_data = [{"id": i+1, "triage": "Red"} for i in range(len(patients))]
        await asyncio.to_thread(allocator.allocate_resources, patients_data)
        return {"message": "Resources allocated successfully."}
    except Exception as e:
        logger.error(f"Error allocating resources: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")