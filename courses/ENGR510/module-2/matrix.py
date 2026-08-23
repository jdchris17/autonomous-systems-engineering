"""A simple, arbitrary-size matrix class for linear algebra exercises."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

# Allow importing Vector from sibling folder: ../module-1/vector.py
MODULE_1_DIR = Path(__file__).resolve().parents[1] / "module-1"
if str(MODULE_1_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_1_DIR))

from vector import Vector


class Matrix:
    def __init__(self, rows):
        rows = [list(row) for row in rows]
        if not rows or not rows[0]:
            raise ValueError("matrix must have at least one row and one column")
        row_lengths = {len(row) for row in rows}
        if len(row_lengths) != 1:
            raise ValueError("all rows must have the same length")
        self.rows = rows

    # -- shape / dimension checking ---------------------------------------

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    @property
    def num_cols(self) -> int:
        return len(self.rows[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.num_rows, self.num_cols)

    @property
    def is_square(self) -> bool:
        return self.num_rows == self.num_cols

    def _check_same_shape(self, other: "Matrix", op: str) -> None:
        if self.shape != other.shape:
            raise ValueError(
                f"cannot {op} a {self.shape[0]}x{self.shape[1]} matrix "
                f"and a {other.shape[0]}x{other.shape[1]} matrix"
            )

    # -- construction helpers ----------------------------------------------

    @staticmethod
    def identity(n: int) -> "Matrix":
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])

    @staticmethod
    def random(num_rows: int, num_cols: int, low: float = 0.0, high: float = 1.0) -> "Matrix":
        return Matrix(
            [[random.uniform(low, high) for _ in range(num_cols)] for _ in range(num_rows)]
        )

    @staticmethod
    def rotation_2d(angle: float, degrees: bool = False) -> "Matrix":
        theta = math.radians(angle) if degrees else angle
        c, s = math.cos(theta), math.sin(theta)
        return Matrix([[c, -s], [s, c]])

    # -- arithmetic ---------------------------------------------------------

    def __add__(self, other: "Matrix") -> "Matrix":
        self._check_same_shape(other, "add")
        return Matrix(
            [[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(self.rows, other.rows)]
        )

    def __sub__(self, other: "Matrix") -> "Matrix":
        self._check_same_shape(other, "subtract")
        return Matrix(
            [[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(self.rows, other.rows)]
        )

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Matrix([[value * other for value in row] for row in self.rows])
        if isinstance(other, Matrix):
            return self._matmul(other)
        if isinstance(other, (Vector, list, tuple)):
            return self.multiply_vector(other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return self * other
        return NotImplemented

    def __pow__(self, exponent: int) -> "Matrix":
        if not self.is_square:
            raise ValueError(f"cannot raise a non-square {self.shape} matrix to a power")
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("matrix power requires a non-negative integer exponent")
        result = Matrix.identity(self.num_rows)
        base = self
        for _ in range(exponent):
            result = result * base
        return result

    def _matmul(self, other: "Matrix") -> "Matrix":
        if self.num_cols != other.num_rows:
            raise ValueError(
                f"cannot multiply a {self.shape[0]}x{self.shape[1]} matrix "
                f"by a {other.shape[0]}x{other.shape[1]} matrix"
            )
        result = []
        for i in range(self.num_rows):
            row = []
            for j in range(other.num_cols):
                total = sum(self.rows[i][k] * other.rows[k][j] for k in range(self.num_cols))
                row.append(total)
            result.append(row)
        return Matrix(result)

    def multiply_vector(self, vector):
        is_vector_type = isinstance(vector, Vector)
        components = [vector.x, vector.y, vector.z] if is_vector_type else list(vector)

        if self.num_cols != len(components):
            raise ValueError(
                f"cannot multiply a {self.shape[0]}x{self.shape[1]} matrix "
                f"by a vector of length {len(components)}"
            )

        result = [
            sum(self.rows[i][k] * components[k] for k in range(self.num_cols))
            for i in range(self.num_rows)
        ]

        if is_vector_type and len(result) == 3:
            return Vector(*result)
        return result

    # -- transformations ------------------------------------------------------

    def transpose(self) -> "Matrix":
        return Matrix([list(col) for col in zip(*self.rows)])

    # -- comparisons & representation -------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix) or self.shape != other.shape:
            return False
        return all(
            math.isclose(a, b, abs_tol=1e-9)
            for r1, r2 in zip(self.rows, other.rows)
            for a, b in zip(r1, r2)
        )

    def __str__(self) -> str:
        return "\n".join(str(row) for row in self.rows)

    def __repr__(self) -> str:
        return f"Matrix({self.rows!r})"

    def pretty_print(self) -> str:
        col_widths = [
            max(len(f"{self.rows[r][c]:g}") for r in range(self.num_rows))
            for c in range(self.num_cols)
        ]
        lines = []
        for row in self.rows:
            cells = [f"{value:g}".rjust(col_widths[c]) for c, value in enumerate(row)]
            lines.append("[ " + "  ".join(cells) + " ]")
        return "\n".join(lines)


if __name__ == "__main__":
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])

    print("a =")
    print(a.pretty_print())
    print("b =")
    print(b.pretty_print())

    print(f"a + b =\n{(a + b).pretty_print()}")
    print(f"a - b =\n{(a - b).pretty_print()}")
    print(f"a * 2 =\n{(a * 2).pretty_print()}")
    print(f"a * b =\n{(a * b).pretty_print()}")
    print(f"b * a =\n{(b * a).pretty_print()}")
    print(f"a transposed =\n{a.transpose().pretty_print()}")
    print(f"I(2) =\n{Matrix.identity(2).pretty_print()}")

    v = Vector(1, 2, 3)
    m3 = Matrix.identity(3) * 2
    print(f"(2 * I3) * {v} = {m3.multiply_vector(v)}")

    rot90 = Matrix.rotation_2d(90, degrees=True)
    print(f"rotation_2d(90deg) * [1, 0] = {rot90.multiply_vector([1, 0])}")
