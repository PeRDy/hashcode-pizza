from collections import Counter
from random import randint

import numpy as np

from hashcode_pizza.genetic import Individual, Population


class Slice:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self._top, self._bottom, self._left, self._right = [None] * 4
        self._size = None
        self._matrix = None

    @staticmethod
    def random(pizza):
        valid = False
        while not valid:
            # Generate first point
            xo = randint(0, pizza.rows - 1)
            yo = randint(0, pizza.cols - 1)

            # Generate second point
            xf = randint(xo, min(xo + pizza.cells, pizza.rows - 1))
            x_size = xf - xo + 1
            y_size = pizza.cells // x_size
            yf = randint(yo, min(yo + y_size, pizza.cols - 1))

            # Create slice
            s = Slice((xo, yo), (xf, yf))

            # Check if valid
            valid = s.is_valid(pizza)

        return s

    @property
    def top(self):
        if self._top is None:
            self._top = max(self.a[1], self.b[1])

        return self._top

    @property
    def bottom(self):
        if self._bottom is None:
            self._bottom = min(self.a[1], self.b[1])

        return self._bottom

    @property
    def left(self):
        if self._left is None:
            self._left = min(self.a[0], self.b[0])

        return self._left

    @property
    def right(self):
        if self._right is None:
            self._right = max(self.a[0], self.b[0])

        return self._right

    def overlaps(self, other: 'Slice') -> bool:
        if self.left > other.right or self.right < other.left:
            return False

        if self.top < other.bottom or self.bottom > other.top:
            return False

        return True

    @property
    def size(self):
        if self._size is None:
            self._size = (abs(self.a[0] - self.b[0]) + 1) * (abs(self.a[1] - self.b[1]) + 1)

        return self._size

    def matrix(self, rows, cols) -> np.array:
        assert rows > max(self.a[0], self.b[0]) and cols > max(self.a[1], self.b[1]), 'Slice out of bounds'

        if self._matrix is None:
            self._matrix = np.zeros((rows, cols))
            xo, xf = min(self.a[0], self.b[0]), max(self.a[0], self.b[0]) + 1
            yo, yf = min(self.a[1], self.b[1]), max(self.a[1], self.b[1]) + 1
            self._matrix[xo:xf, yo:yf] = 1

        return self._matrix

    def is_valid(self, pizza) -> bool:
        try:
            inverted_slice = 1 - self.matrix(pizza.rows, pizza.cols)
        except AssertionError:
            valid = False
        else:
            slice_mask = np.ma.array(pizza.pizza, mask=inverted_slice)
            enough_ingredients = {k for k, v in Counter(slice_mask[~slice_mask.mask]).items() if v >= pizza.ingredients}
            valid = self.size <= pizza.cells and len(enough_ingredients) == 2

        return valid

    def __str__(self):
        return f'{self.a[0]} {self.b[0]} {self.a[1]} {self.b[1]}'

    def __repr__(self):
        return f'Slice{{{self.a}, {self.b}, size={self.size}}}'


class Pizza:
    def __init__(self, rows, cols, ingredients, cells, pizza):
        self.rows = rows
        self.cols = cols
        self.ingredients = ingredients
        self.cells = cells
        self.pizza = pizza
        self._size = None

    @property
    def size(self):
        if self._size is None:
            self._size = self.rows * self.cols

        return self._size

    def __str__(self):
        return '\n'.join([''.join(i) for i in self.pizza])

    def __repr__(self):
        return f'Pizza{{rows={self.rows}, cols={self.cols}, ingredients={self.ingredients}, cells={self.cells}, ' \
               f'size={self.size}}}\n' \
               f'{str(self)}'


class Solution(Individual):
    def __init__(self, pizza, slices):
        self.pizza = pizza
        self.slices = slices

    @staticmethod
    def random(pizza, stability=0.2, nth=3):
        solution = Solution(pizza, [])
        fitness = []
        # stable = False
        # while not stable:
        for i in range(nth):
            solution.mutate()
        #     fitness.append(solution.fitness)
        #     stable = len(fitness) >= nth and np.std(fitness[len(fitness) - nth:], ddof=1) < stability

        return solution

    def mutate(self):
        new_slice = Slice.random(self.pizza)
        self.slices = [s for s in self.slices if not new_slice.overlaps(s)] + [new_slice]

    def breed(self, mother: 'Solution') -> 'Solution':
        father_slices = [s for s in self.slices if s.right < self.pizza.cols // 2]
        mother_slices = [s for s in mother.slices if s.left >= self.pizza.cols // 2]

        return Solution(pizza=self.pizza, slices=father_slices + mother_slices)

    @property
    def fitness(self) -> float:
        return sum([s.size for s in self.slices]) / self.pizza.size

    def __str__(self):
        slices = '\n'.join([str(s) for s in self.slices])
        return f'{len(self.slices)}\n{slices}'

    def __repr__(self):
        r = f'Solution{{fitness={self.fitness}, slices={len(self.slices)}}}'
        if self.pizza.cols <= 10 and self.pizza.rows <= 10:
            matrix = sum([s.matrix(self.pizza.rows, self.pizza.cols) * (i + 1) for i, s in enumerate(self.slices)])
            matrix_repr = '\n'.join([' '.join([f'{j:2.0f}' for j in i]) for i in matrix])
            r += f'\n{matrix_repr}'
        return r


class SolutionSet(Population):
    @classmethod
    def read(cls, file_path, population: int, *args, **kwargs):
        with open(file_path) as f:
            rows, cols, min_ingredients, max_cells = [int(i) for i in f.readline().split()]
            pizza_data = np.array([list(i.strip()) for i in f.readlines()])

        pizza = Pizza(rows, cols, min_ingredients, max_cells, pizza_data)
        return cls(pizza, population)

    def write(self, file_path, *args, **kwargs):
        with open(file_path, 'w') as f:
            f.write(str(self.best))

    def __init__(self, pizza, size, *args, **kwargs):
        self.pizza = pizza
        self.individuals = [Solution.random(pizza) for _ in range(size)]
