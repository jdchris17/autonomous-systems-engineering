"""A simple 3D vector class for linear algebra exercises."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Vector:
    x: float
    y: float
    z: float = 0.0

    # -- arithmetic -----------------------------------------------------

    def __add__(self, other: "Vector") -> "Vector":
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector") -> "Vector":
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vector":
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Vector":
        if scalar == 0:
            raise ZeroDivisionError("cannot divide a vector by zero")
        return Vector(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vector":
        return Vector(-self.x, -self.y, -self.z)

    # -- core operations --------------------------------------------------

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self) -> "Vector":
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("cannot normalize the zero vector")
        return self / mag

    def dot(self, other: "Vector") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector") -> "Vector":
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def angle_between(self, other: "Vector", degrees: bool = False) -> float:
        mag_product = self.magnitude() * other.magnitude()
        if mag_product == 0:
            raise ValueError("cannot compute angle with the zero vector")
        # clamp to [-1, 1] to guard against floating point drift
        cos_theta = max(-1.0, min(1.0, self.dot(other) / mag_product))
        angle = math.acos(cos_theta)
        return math.degrees(angle) if degrees else angle

    def distance_to(self, other: "Vector") -> float:
        return (self - other).magnitude()

    def project_onto(self, other: "Vector") -> "Vector":
        other_mag_sq = other.dot(other)
        if other_mag_sq == 0:
            raise ValueError("cannot project onto the zero vector")
        scalar = self.dot(other) / other_mag_sq
        return other * scalar

    # -- comparisons & representation -------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return (
            math.isclose(self.x, other.x, abs_tol=1e-9)
            and math.isclose(self.y, other.y, abs_tol=1e-9)
            and math.isclose(self.z, other.z, abs_tol=1e-9)
        )

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z})"

    def __repr__(self) -> str:
        return f"Vector({self.x!r}, {self.y!r}, {self.z!r})"


if __name__ == "__main__":
    a = Vector(1, 2, 3)
    b = Vector(4, 5, 6)

    print(f"a = {a}")
    print(f"b = {b}")

    print(f"a + b = {a + b}")
    print(f"a - b = {a - b}")
    print(f"a * 2 = {a * 2}")
    print(f"a / 2 = {a / 2}")
    print(f"-a = {-a}")

    print(f"|a| = {a.magnitude():.4f}")
    print(f"a normalized = {a.normalize()}")
    print(f"|a normalized| = {a.normalize().magnitude():.4f}")

    print(f"a . b = {a.dot(b)}")
    print(f"a x b = {a.cross(b)}")
    print(f"angle(a, b) = {a.angle_between(b, degrees=True):.2f} degrees")
    print(f"distance(a, b) = {a.distance_to(b):.4f}")
    print(f"projection of a onto b = {a.project_onto(b)}")

    print(f"a == b? {a == b}")
    print(f"a == Vector(1, 2, 3)? {a == Vector(1, 2, 3)}")

    # orthogonal example
    i, j = Vector(1, 0, 0), Vector(0, 1, 0)
    print(f"i . j = {i.dot(j)} (orthogonal -> 0)")

    # parallel example
    c = Vector(2, 4, 6)
    print(f"angle(a, c) = {a.angle_between(c, degrees=True):.2f} degrees (parallel -> 0)")
