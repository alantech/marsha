import unittest
import fibonacci from fibonacci


class TestFibonacci(unittest.TestCase):

  def test_1(self):
    self.assertEqual(fibonacci(1), 1)

  def test_2(self):
    self.assertEqual(fibonacci(2), 1)

  def test_3(self):
    self.assertEqual(fibonacci(3), 2)

  def test_0(self):
    self.assertRaises(Exception, fibonacci, 0)


if __name__ == '__main__':
    unittest.main()
