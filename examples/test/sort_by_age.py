from typing import List


class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Person(name={self.name}, age={self.age})"

    def __str__(self):
        return f"Person(name={self.name}, age={self.age})"

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age


def sort_by_age(people: List[Person], ascending: bool = True) -> List[Person]:
    """
    This function receives a list of `Person` objects and returns them ordered by age, 
    either in ascending or descending order depending on the boolean flag `ascending`.

    Args:
      - people: A list of `Person` objects.
      - ascending: A boolean flag indicating the ordering (default is True for ascending order).

    Returns:
      - A list of `Person` objects ordered by age.

    Raises:
      - TypeError: If `people` is not a list.
    """
    if not isinstance(people, list):
        raise TypeError(
            "The 'people' argument should be a list of Person objects.")

    sorted_people = sorted(people, key=lambda p: p.age)
    return sorted_people if ascending else sorted_people[::-1]
