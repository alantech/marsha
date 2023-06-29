import unittest
from get_employee_skills import get_employee_skills, Employee, DepartmentSkills, EmployeeSkills


class TestGetEmployeeSkills(unittest.TestCase):

    def test_no_args(self):
        self.assertRaises(TypeError, get_employee_skills)

    def test_empty_lists(self):
        self.assertEqual(get_employee_skills([], []), [])

    def test_empty_employee_list(self):
        self.assertEqual(get_employee_skills(
            [], [DepartmentSkills('Accounting', 'math')]), [])

    def test_empty_skills_list(self):
        self.assertEqual(get_employee_skills(
            [Employee('Joe', 'Accounting')], []), [])

    def test_single_department_single_employee(self):
        employees = [Employee('Joe', 'Accounting')]
        department_skills = [DepartmentSkills('Accounting', 'math')]

        expected_results = [EmployeeSkills('Joe', 'math')]

        self.assertEqual(get_employee_skills(
            employees, department_skills), expected_results)

    def test_multiple_departments_single_employee(self):
        employees = [Employee('Joe', 'Accounting'),
                     Employee('Jake', 'Engineering')]
        department_skills = [DepartmentSkills('Accounting', 'math')]

        expected_results = [EmployeeSkills('Joe', 'math')]

        self.assertEqual(get_employee_skills(
            employees, department_skills), expected_results)

    def test_multiple_departments_multiple_employees(self):
        employees = [Employee('Joe', 'Accounting'),
                     Employee('Jake', 'Engineering')]
        department_skills = [DepartmentSkills(
            'Accounting', 'math'), DepartmentSkills('Engineering', 'coding')]

        expected_results = [EmployeeSkills(
            'Joe', 'math'), EmployeeSkills('Jake', 'coding')]

        self.assertEqual(get_employee_skills(
            employees, department_skills), expected_results)


if __name__ == '__main__':
    unittest.main()
