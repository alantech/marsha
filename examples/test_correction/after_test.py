import unittest
from extract_connection_info import extract_connection_info


class TestExtractConnectionInfo(unittest.TestCase):

    def test_basic_postgresql(self):
        expected = {"protocol": "postgresql", "dbUser": "user", "dbPassword": "pass", "host": "0.0.0.0", "port": 5432, "database": "mydb"}
        result = extract_connection_info('postgresql://user:pass@0.0.0.0:5432/mydb')
        self.assertEqual(result, expected)

    def test_postgresql_with_extra(self):
        expected = {"protocol": "postgresql", "dbUser": "user", "dbPassword": "pass", "host": "0.0.0.0", "port": 5432, "database": "mydb", "extra": {"ssl": "require"}}
        result = extract_connection_info('postgresql://user:pass@0.0.0.0:5432/mydb?sslmode=require')
        self.assertEqual(result, expected)

    def test_basic_mysql(self):
        expected = {"protocol": "mysql", "dbUser": "user", "dbPassword": "pass", "host": "0.0.0.0", "port": 3306, "database": "mydb"}
        result = extract_connection_info('jdbc:mysql://0.0.0.0:3306/mydb?user=user&password=pass')
        self.assertEqual(result, expected)

    def test_mysql_with_extra(self):
        expected = {"protocol": "mysql", "dbUser": "user", "dbPassword": "pass", "host": "0.0.0.0", "port": 3306, "database": "mydb", "extra": {"ssl": "require"}}
        result = extract_connection_info('jdbc:mysql://0.0.0.0:3306/mydb?user=user&password=pass&sslMode=REQUIRED')
        self.assertEqual(result, expected)

    def test_empty_string(self):
        expected = {}
        result = extract_connection_info('')
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
