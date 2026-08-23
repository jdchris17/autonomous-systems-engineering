import math

import pytest

from vector import Vector


# -- construction / representation ---------------------------------------

def test_string_representation():
    v = Vector(1, 2, 3)
    assert str(v) == "(1, 2, 3)"
    assert repr(v) == "Vector(1, 2, 3)"


# -- addition / subtraction -----------------------------------------------

def test_addition():
    assert Vector(1, 2, 3) + Vector(4, 5, 6) == Vector(5, 7, 9)


def test_subtraction():
    assert Vector(4, 5, 6) - Vector(1, 2, 3) == Vector(3, 3, 3)


# -- scalar multiplication / division --------------------------------------

def test_scalar_multiplication():
    assert Vector(1, 2, 3) * 2 == Vector(2, 4, 6)
    assert 2 * Vector(1, 2, 3) == Vector(2, 4, 6)


def test_scalar_division():
    assert Vector(2, 4, 6) / 2 == Vector(1, 2, 3)


def test_scalar_division_by_zero_raises():
    with pytest.raises(ZeroDivisionError):
        Vector(1, 2, 3) / 0


# -- magnitude / normalization ---------------------------------------------

def test_magnitude():
    assert Vector(3, 4, 0).magnitude() == 5


def test_normalize_returns_unit_vector():
    v = Vector(3, 4, 0).normalize()
    assert math.isclose(v.magnitude(), 1.0)
    assert v == Vector(0.6, 0.8, 0.0)


def test_normalize_zero_vector_raises():
    with pytest.raises(ValueError):
        Vector(0, 0, 0).normalize()


# -- dot product / angle / distance ----------------------------------------

def test_dot_product():
    assert Vector(1, 2, 3).dot(Vector(4, 5, 6)) == 32


def test_angle_between_raises_for_zero_vector():
    with pytest.raises(ValueError):
        Vector(0, 0, 0).angle_between(Vector(1, 0, 0))


def test_distance_between():
    assert Vector(0, 0, 0).distance_to(Vector(3, 4, 0)) == 5


# -- equality ---------------------------------------------------------------

def test_equality_and_inequality():
    assert Vector(1, 2, 3) == Vector(1, 2, 3)
    assert Vector(1, 2, 3) != Vector(1, 2, 4)


# -- stretch goals: cross product / projection -----------------------------

def test_cross_product():
    i, j, k = Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)
    assert i.cross(j) == k
    assert j.cross(i) == -k


def test_projection_onto_axis():
    v = Vector(3, 4, 0)
    onto_x = v.project_onto(Vector(1, 0, 0))
    assert onto_x == Vector(3, 0, 0)


def test_projection_onto_zero_vector_raises():
    with pytest.raises(ValueError):
        Vector(1, 2, 3).project_onto(Vector(0, 0, 0))


# -- required conceptual test cases ------------------------------------------

class TestOrthogonalVectors:
    def test_dot_product_is_zero(self):
        assert Vector(1, 0, 0).dot(Vector(0, 1, 0)) == 0

    def test_angle_is_ninety_degrees(self):
        angle = Vector(1, 0, 0).angle_between(Vector(0, 1, 0), degrees=True)
        assert math.isclose(angle, 90.0)


class TestParallelVectors:
    def test_same_direction_angle_is_zero(self):
        angle = Vector(1, 2, 3).angle_between(Vector(2, 4, 6))
        assert math.isclose(angle, 0.0, abs_tol=1e-9)

    def test_opposite_direction_angle_is_180(self):
        angle = Vector(1, 2, 3).angle_between(Vector(-1, -2, -3), degrees=True)
        assert math.isclose(angle, 180.0)

    def test_cross_product_of_parallel_vectors_is_zero(self):
        assert Vector(1, 2, 3).cross(Vector(2, 4, 6)) == Vector(0, 0, 0)


class TestUnitVectors:
    def test_standard_basis_vectors_have_magnitude_one(self):
        assert Vector(1, 0, 0).magnitude() == 1
        assert Vector(0, 1, 0).magnitude() == 1
        assert Vector(0, 0, 1).magnitude() == 1

    def test_normalizing_a_unit_vector_is_a_no_op(self):
        v = Vector(1, 0, 0)
        assert v.normalize() == v


class TestZeroVector:
    def test_magnitude_is_zero(self):
        assert Vector(0, 0, 0).magnitude() == 0

    def test_addition_is_identity(self):
        v = Vector(3, -1, 2)
        assert v + Vector(0, 0, 0) == v

    def test_normalize_raises(self):
        with pytest.raises(ValueError):
            Vector(0, 0, 0).normalize()

    def test_angle_between_raises(self):
        with pytest.raises(ValueError):
            Vector(0, 0, 0).angle_between(Vector(1, 1, 1))


class TestNegativeVectors:
    def test_negation(self):
        assert -Vector(1, -2, 3) == Vector(-1, 2, -3)

    def test_magnitude_ignores_sign(self):
        assert Vector(-3, -4, 0).magnitude() == 5

    def test_dot_product_with_negation_is_negative_magnitude_squared(self):
        v = Vector(1, 2, 3)
        assert v.dot(-v) == -v.dot(v)

    def test_subtraction_with_negatives(self):
        assert Vector(-1, -2, -3) - Vector(1, 2, 3) == Vector(-2, -4, -6)


class TestFloatingPointComparisons:
    def test_accumulated_rounding_error_still_compares_equal(self):
        # 0.1 + 0.2 != 0.3 exactly in binary floating point
        v = Vector(0.1 + 0.2, 0, 0)
        assert v == Vector(0.3, 0, 0)

    def test_normalized_vector_magnitude_is_close_to_one(self):
        v = Vector(1, 1, 1).normalize()
        assert math.isclose(v.magnitude(), 1.0, rel_tol=1e-9)

    def test_tiny_difference_beyond_tolerance_is_not_equal(self):
        assert Vector(1, 0, 0) != Vector(1.1, 0, 0)
