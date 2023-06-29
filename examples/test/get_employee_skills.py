from typing import List
import pandas as pd


class Employee:
    def __init__(self, name: str, department: str):
        self.name = name
        self.department = department

    def __repr__(self):
        return f'Employee(name={self.name}, department={self.department})'

    def __str__(self):
        return f'Employee(name={self.name}, department={self.department})'

    def __eq__(self, other):
        return self.name == other.name and self.department == other.department


class DepartmentSkills:
    def __init__(self, department: str, skill: str):
        self.department = department
        self.skill = skill

    def __repr__(self):
        return f'DepartmentSkills(department={self.department}, skill={self.skill})'

    def __str__(self):
        return f'DepartmentSkills(department={self.department}, skill={self.skill})'

    def __eq__(self, other):
        return self.department == other.department and self.skill == other.skill


class EmployeeSkills:
    def __init__(self, name: str, skill: str):
        self.name = name
        self.skill = skill

    def __repr__(self):
        return f'EmployeeSkills(name={self.name}, skill={self.skill})'

    def __str__(self):
        return f'EmployeeSkills(name={self.name}, skill={self.skill})'

    def __eq__(self, other):
        return self.name == other.name and self.skill == other.skill


def get_employee_skills(employees: List[Employee], department_skills: List[DepartmentSkills]) -> List[EmployeeSkills]:

    if not employees or not department_skills:
        return []

    employees_df = pd.DataFrame([(employee.name, employee.department)
                                for employee in employees], columns=['name', 'department'])
    skills_df = pd.DataFrame([(skill.department, skill.skill)
                             for skill in department_skills], columns=['department', 'skill'])

    merged_df = employees_df.merge(skills_df, on='department')
    employee_skills = [EmployeeSkills(
        row['name'], row['skill']) for _, row in merged_df.iterrows()]

    return employee_skills
