def fibonacci(n: int) -> int:
  """This function calculates the nth fibonacci number, where n is provided to it and starts with 1.

  fibonacci(n) = fibonacci(n - 1) + fibonacci(n - 2)"""
  if n <= 0:
    raise Exception('The fibonacci sequence only exists in positive whole number space')
  elif n == 1 or n == 2:
    return 1
  else:
    return fibonacci(n - 1) + fibonacci(n - 2)
