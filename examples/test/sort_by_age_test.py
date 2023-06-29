import unittest
from sort_by_age import Person, sort_by_age


class TestSortByAge(unittest.TestCase):

    def test_sort_ascending(self):
        people = [Person('Joe', 20), Person('Jane', 50),
                  Person('Felix', 10), Person('Alex', 60)]
        expected = [Person('Felix', 10), Person('Joe', 20),
                    Person('Jane', 50), Person('Alex', 60)]
        self.assertEqual(sort_by_age(people), expected)

    def test_sort_descending(self):
        people = [Person('Joe', 20), Person('Jane', 50),
                  Person('Felix', 10), Person('Alex', 60)]
        expected = [Person('Alex', 60), Person('Jane', 50),
                    Person('Joe', 20), Person('Felix', 10)]
        self.assertEqual(sort_by_age(people, False), expected)

    def test_sort_empty_list(self):
        people = []
        expected = []
        self.assertEqual(sort_by_age(people), expected)

    def test_type_error_no_list_received(self):
        self.assertRaises(TypeError, sort_by_age, 'not a list')

    def test_type_error_no_list_received_no_args(self):
        self.assertRaises(TypeError, sort_by_age)


if __name__ == '__main__':
    unittest.main()
