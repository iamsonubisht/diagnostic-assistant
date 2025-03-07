from ortools.linear_solver import pywraplp
import asyncio
import logging

logger = logging.getLogger(__name__)

class HospitalResourceAllocator:
    def __init__(self, hospitals, patients):
        self.hospitals = hospitals
        self.patients = patients
        
        # 🔄 Ensure correct key name
        for patient in self.patients.values():
            patient['triage'] = patient.pop('triage_category', 'Green')  # ✅ Rename safely

    async def allocate_resources(self):  # ✅ Correct indentation
        solver = pywraplp.Solver.CreateSolver('SCIP')
        hospital_names = list(self.hospitals.keys())

        # 🔹 Fix patient ID mapping
        patient_ids = {details['id']: patient_name for patient_name, details in self.patients.items()}

        # Decision variables
        x = {(i, j): solver.IntVar(0, 1, f'x_{i}_{j}') for i in patient_ids.keys() for j in hospital_names}

        # Constraints
        for i in patient_ids.keys():
            solver.Add(sum(x[(i, j)] for j in hospital_names) == 1)

        for j in hospital_names:
            solver.Add(sum(x[(i, j)] for i in patient_ids.keys()) <= self.hospitals[j]['capacity'])
            solver.Add(sum(x[(i, j)] for i in patient_ids.keys() if self.patients[patient_ids[i]]['triage'] == 'Red') <= self.hospitals[j].get('ambulances', 0))

        # Objective
        objective = solver.Objective()
        for i in patient_ids.keys():
            for j in hospital_names:
                patient_name = patient_ids[i]
                triage = self.patients[patient_name]['triage']  # ✅ Using renamed key
                priority = 10 if triage == 'Red' else (5 if triage == 'Yellow' else 1)
                objective.SetCoefficient(x[(i, j)], priority)
        objective.SetMaximization()

        # Solve
        status = solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            logger.info("Optimal resource allocation found.")
            for i in patient_ids.keys():
                for j in hospital_names:
                    if x[(i, j)].solution_value() == 1:
                        print(f"Patient {i} (Triage: {self.patients[patient_ids[i]]['triage']}) -> {j}")
        else:
            logger.warning("No optimal solution found.")
