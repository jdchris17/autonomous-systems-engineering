import math

import pytest

from matrix import Matrix
from vector import Vector


# -- construction / representation ---------------------------------------

def test_matrix_creation_and_shape():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    assert m.shape == (2, 3)
    assert m.rows == [[1, 2, 3], [4, 5, 6]]


def test_ragged_rows_raise():
    with pytest.raises(ValueError):
        Matrix([[1, 2], [3, 4, 5]])


def test_string_representation():
    m = Matrix([[1, 2], [3, 4]])
    assert str(m) == "[1, 2]\n[3, 4]"
    assert repr(m) == "Matrix([[1, 2], [3, 4]])"


def test_equality():
    assert Matrix([[1, 2], [3, 4]]) == Matrix([[1, 2], [3, 4]])
    assert Matrix([[1, 2], [3, 4]]) != Matrix([[1, 2], [3, 5]])
    assert Matrix([[1, 2], [3, 4]]) != Matrix([[1, 2, 3], [4, 5, 6]])


# -- addition / subtraction / scalar multiplication ------------------------

def test_addition():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    assert a + b == Matrix([[6, 8], [10, 12]])


def test_subtraction():
    a = Matrix([[5, 6], [7, 8]])
    b = Matrix([[1, 2], [3, 4]])
    assert a - b == Matrix([[4, 4], [4, 4]])


def test_scalar_multiplication():
    a = Matrix([[1, 2], [3, 4]])
    assert a * 2 == Matrix([[2, 4], [6, 8]])
    assert 2 * a == Matrix([[2, 4], [6, 8]])


# -- matrix multiplication --------------------------------------------------

def test_matrix_multiplication():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    assert a * b == Matrix([[19, 22], [43, 50]])


def test_matrix_multiplication_non_square_dimensions():
    a = Matrix([[1, 2, 3], [4, 5, 6]])   # 2x3
    b = Matrix([[7, 8], [9, 10], [11, 12]])  # 3x2
    assert a * b == Matrix([[58, 64], [139, 154]])  # 2x2


# -- transpose ---------------------------------------------------------------

def test_transpose():
    a = Matrix([[1, 2, 3], [4, 5, 6]])
    assert a.transpose() == Matrix([[1, 4], [2, 5], [3, 6]])
    assert a.transpose().shape == (3, 2)


# -- stretch: matrix power, random matrix ------------------------------------

def test_matrix_power():
    a = Matrix([[1, 1], [0, 1]])
    assert a ** 0 == Matrix.identity(2)
    assert a ** 1 == a
    assert a ** 3 == Matrix([[1, 3], [0, 1]])


def test_matrix_power_requires_square():
    with pytest.raises(ValueError):
        Matrix([[1, 2, 3], [4, 5, 6]]) ** 2


def test_random_matrix_has_requested_shape_and_bounds():
    m = Matrix.random(3, 4, low=-1.0, high=1.0)
    assert m.shape == (3, 4)
    assert all(-1.0 <= value <= 1.0 for row in m.rows for value in row)


# ============================================================
# Required conceptual test cases
# ============================================================

class TestIdentityMatrixBehavior:
    def test_identity_times_matrix_is_matrix(self):
        a = Matrix([[1, 2], [3, 4]])
        assert Matrix.identity(2) * a == a

    def test_matrix_times_identity_is_matrix(self):
        a = Matrix([[1, 2], [3, 4]])
        assert a * Matrix.identity(2) == a

    def test_identity_diagonal_is_ones(self):
        i3 = Matrix.identity(3)
        assert i3 == Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    def test_identity_times_vector_is_unchanged(self):
        v = Vector(7, -2, 5)
        assert Matrix.identity(3).multiply_vector(v) == v


class TestAssociativityOfMultiplication:
    def test_matrix_multiplication_is_associative(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[2, 0], [1, 3]])
        c = Matrix([[0, 1], [4, 2]])
        assert (a * b) * c == a * (b * c)

    def test_associative_with_non_square_matrices(self):
        a = Matrix([[1, 2, 3], [4, 5, 6]])       # 2x3
        b = Matrix([[1, 0], [0, 1], [1, 1]])     # 3x2
        c = Matrix([[2, 0], [0, 2]])             # 2x2
        assert (a * b) * c == a * (b * c)


class TestNonCommutativity:
    def test_matrix_multiplication_is_not_commutative_in_general(self):
        a = Matrix([[1, 2], [3, 4]])
        b = Matrix([[5, 6], [7, 8]])
        assert a * b != b * a

    def test_identity_is_the_exception_that_commutes(self):
        a = Matrix([[1, 2], [3, 4]])
        i = Matrix.identity(2)
        assert a * i == i * a


class TestMatrixVectorMultiplication:
    def test_multiply_vector_class_instance(self):
        m = Matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        v = Vector(1, 1, 1)
        assert m.multiply_vector(v) == Vector(1, 2, 3)

    def test_multiply_returns_vector_type_for_3x3(self):
        m = Matrix.identity(3)
        result = m.multiply_vector(Vector(1, 2, 3))
        assert isinstance(result, Vector)

    def test_multiply_plain_list(self):
        m = Matrix([[2, 0], [0, 2]])
        assert m.multiply_vector([3, 4]) == [6, 8]

    def test_mul_operator_dispatches_to_vector_multiplication(self):
        m = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert m * Vector(4, 5, 6) == Vector(4, 5, 6)


class TestDimensionMismatchErrors:
    def test_addition_mismatch_raises(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2], [3, 4]]) + Matrix([[1, 2, 3], [4, 5, 6]])

    def test_subtraction_mismatch_raises(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2], [3, 4]]) - Matrix([[1, 2, 3], [4, 5, 6]])

    def test_matrix_multiplication_inner_dimension_mismatch_raises(self):
        a = Matrix([[1, 2], [3, 4]])  # 2x2, needs 2 rows on the right-hand side
        b = Matrix([[1, 2, 3]])       # 1x3
        with pytest.raises(ValueError):
            a * b

    def test_matrix_vector_multiplication_length_mismatch_raises(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])  # 2x3
        with pytest.raises(ValueError):
            m.multiply_vector([1, 2])  # length 2, needs length 3

    def test_ragged_rows_raise_on_construction(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2, 3], [4, 5]])

    def test_non_square_matrix_power_raises(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2, 3], [4, 5, 6]]) ** 2


class TestRotationMatrixCorrectness:
    def test_rotation_90_degrees_maps_x_axis_to_y_axis(self):
        rot90 = Matrix.rotation_2d(90, degrees=True)
        result = rot90.multiply_vector([1, 0])
        assert math.isclose(result[0], 0.0, abs_tol=1e-9)
        assert math.isclose(result[1], 1.0, abs_tol=1e-9)

    def test_rotation_180_degrees_negates_vector(self):
        rot180 = Matrix.rotation_2d(180, degrees=True)
        result = rot180.multiply_vector([2, 3])
        assert math.isclose(result[0], -2.0, abs_tol=1e-9)
        assert math.isclose(result[1], -3.0, abs_tol=1e-9)

    def test_rotation_preserves_vector_length(self):
        rot37 = Matrix.rotation_2d(37, degrees=True)
        result = rot37.multiply_vector([3, 4])
        length = math.hypot(*result)
        assert math.isclose(length, 5.0, abs_tol=1e-9)

    def test_rotation_by_zero_is_identity(self):
        rot0 = Matrix.rotation_2d(0, degrees=True)
        assert rot0 == Matrix.identity(2)

    def test_two_rotations_compose_by_adding_angles(self):
        rot30 = Matrix.rotation_2d(30, degrees=True)
        rot60 = Matrix.rotation_2d(60, degrees=True)
        rot90 = Matrix.rotation_2d(90, degrees=True)
        assert rot60 * rot30 == rot90
