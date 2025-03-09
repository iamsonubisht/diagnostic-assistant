import asyncio
from ortools.linear_solver import pywraplp
import logging

logger = logging.getLogger(__name__)

class HospitalResourceAllocator:
    def __init__(self, hospitals, patients):
        self.hospitals = hospitals
        self.patients = patients
        
        # Ensure correct key name for triage categories
        for patient in self.patients.values():
            patient['triage'] = patient.pop('triage_category', 'Green')  # Rename safely

    async def allocate_resources(self, predicted_triage):
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            logger.error("Solver not created. Check if OR-Tools is installed correctly.")
            return []

        hospital_names = list(self.hospitals.keys())
        patient_ids = {details['id']: patient_name for patient_name, details in self.patients.items()}

        # Decision variables: x[i, j] = 1 if patient i is assigned to hospital j
        x = {(i, j): solver.IntVar(0, 1, f'x_{i}_{j}') for i in patient_ids.keys() for j in hospital_names}

        # Constraints: Each patient must be assigned to exactly one hospital
        for i in patient_ids.keys():
            solver.Add(sum(x[(i, j)] for j in hospital_names) == 1)

        # Constraints: Hospital capacity and ambulance limits
        for j in hospital_names:
            solver.Add(sum(x[(i, j)] for i in patient_ids.keys()) <= self.hospitals[j]['capacity'])
            solver.Add(sum(x[(i, j)] for i in patient_ids.keys() if self.patients[patient_ids[i]]['triage'] == 'Red') <= self.hospitals[j].get('ambulances', 0))

        # Objective: Maximize the priority of patient allocations
        objective = solver.Objective()
        for i in patient_ids.keys():
            for j in hospital_names:
                triage = predicted_triage[i - 1]  # Use the predicted triage from the input
                priority = self.get_priority(triage)  # Use a method to get priority
                objective.SetCoefficient(x[(i, j)], priority)
        objective.SetMaximization()

        # Solve the optimization problem
        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            logger.info("Optimal resource allocation found.")
            allocations = self.get_allocations(x, patient_ids, hospital_names)
            return allocations
        else:
            logger.warning("No optimal solution found. Status: %s", status)
            return []  # Return an empty list if no solution is found

    def get_priority(self, triage):
        """Return the priority based on triage category."""
        if triage == 'Red':
            return 10
        elif triage == 'Yellow':
            return 5
        else:  # Green
            return 1

    def get_allocations(self, x, patient_ids, hospital_names):
        """Return a detailed allocation of patients to hospitals."""
        allocations = []
        for i in patient_ids.keys():
            for j in hospital_names:
                if x[(i, j)].solution_value() == 1:
                    allocations.append({
                        'patient_id': patient_ids[i],
                        'hospital': j,
                        'triage': self.patients[patient_ids[i]]['triage']
                    })
        return allocations