from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import asyncio
import logging

logger = logging.getLogger(__name__)

Base = declarative_base()

class Patient(Base):
    __tablename__ = 'patients'
    id = Column(Integer, primary_key=True)
    heart_rate = Column(Integer)
    blood_pressure = Column(Integer)
    oxygen_level = Column(Integer)
    injury_severity = Column(Integer)
    triage_category = Column(String)

class Database:
    def __init__(self, db_url):
        self.engine = create_async_engine(db_url)
        self.Session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def save_patient(self, patient_data):
        """Asynchronously save patient data to the database."""
        async with self.Session() as session:
            patient = Patient(**patient_data)
            session.add(patient)
            await session.commit()
            logger.info(f"Patient {patient.id} saved to the database.")