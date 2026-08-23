"""Matrix challenge problems: rotations, scaling, robot kinematics,
and homogeneous coordinates.
"""

import math

from matrix import Matrix


# -- Challenge 1: 45-degree rotation, magnitude preservation ----------------
# Math: a rotation matrix is orthogonal (R^T R = I), which is exactly the
# property that makes it length-preserving. Applying it to a vector changes
# direction only -- magnitude, i.e. sqrt(x^2 + y^2), is invariant.

def challenge_1():
    print("Challenge 1: 45-degree rotation matrix")
    rot45 = Matrix.rotation_2d(45, degrees=True)
    print("  rotation matrix =")
    print(f"  {rot45.pretty_print()}".replace("\n", "\n  "))

    test_vectors = [[1, 0], [0, 1], [3, 4], [-2, 5], [1, 1]]
    for v in test_vectors:
        rotated = rot45.multiply_vector(v)
        before = math.hypot(*v)
        after = math.hypot(*rotated)
        print(
            f"  v={v}  ->  rotated={[round(c, 4) for c in rotated]}  "
            f"|v|={before:.4f}  |rotated|={after:.4f}  "
            f"preserved={math.isclose(before, after, abs_tol=1e-9)}"
        )
    print()


# -- Challenge 2: scaling vs. rotation, order matters ------------------------
# Math: matrix multiplication is not commutative (R @ S != S @ R in
# general), so "scale then rotate" and "rotate then scale" are genuinely
# different transformations, not just different notation for the same one.

def challenge_2():
    print("Challenge 2: scaling and rotation applied in different orders")
    scale = Matrix([[2, 0], [0, 0.5]])          # stretch x by 2, squeeze y by 0.5
    rotate = Matrix.rotation_2d(30, degrees=True)
    v = [1, 1]

    scale_then_rotate = rotate.multiply_vector(scale.multiply_vector(v))
    rotate_then_scale = scale.multiply_vector(rotate.multiply_vector(v))

    print(f"  v = {v}")
    print(f"  scale then rotate: rotate * (scale * v) = {[round(c, 4) for c in scale_then_rotate]}")
    print(f"  rotate then scale: scale * (rotate * v) = {[round(c, 4) for c in rotate_then_scale]}")
    print(
        "  -> different results, because matrix multiplication is not\n"
        "     commutative: scaling first stretches the vector along the\n"
        "     original axes before the rotation reorients it, while\n"
        "     rotating first reorients the vector before the scale is\n"
        "     applied along the (now-fixed) x/y axes. The two operations\n"
        "     only commute when the scale is uniform (sx == sy), because\n"
        "     a uniform scale is just a multiple of the identity matrix,\n"
        "     and cI commutes with every matrix."
    )
    print()


# -- Challenge 3: two-joint planar robot arm ---------------------------------
# Math: each joint is a homogeneous transform combining a rotation (the
# joint angle) with a translation (the link length along the arm's local
# x-axis). Chaining joints is chaining matrix multiplications: the
# end-effector position in the base frame is T1 @ T2 @ [0, 0, 1]^T, where
# [0, 0, 1] is the end-effector's own origin in homogeneous coordinates.

def homogeneous_rotation(degrees: float) -> Matrix:
    r = Matrix.rotation_2d(degrees, degrees=True)
    return Matrix(
        [
            [r.rows[0][0], r.rows[0][1], 0],
            [r.rows[1][0], r.rows[1][1], 0],
            [0, 0, 1],
        ]
    )


def homogeneous_translation(tx: float, ty: float) -> Matrix:
    return Matrix([[1, 0, tx], [0, 1, ty], [0, 0, 1]])


def joint_transform(angle_degrees: float, link_length: float) -> Matrix:
    """Rotate by the joint angle, then translate along the new x-axis
    by the link length -- one homogeneous matrix per joint."""
    return homogeneous_rotation(angle_degrees) * homogeneous_translation(link_length, 0)


def challenge_3():
    print("Challenge 3: two-joint planar robot arm")
    theta1, length1 = 30, 4.0   # shoulder joint: 30 degrees, 4-unit link
    theta2, length2 = 45, 3.0   # elbow joint (relative angle): 45 degrees, 3-unit link

    joint1 = joint_transform(theta1, length1)
    joint2 = joint_transform(theta2, length2)

    origin = [0, 0, 1]  # a point (not a direction) in homogeneous coordinates
    elbow_position = joint1.multiply_vector(origin)
    end_effector = (joint1 * joint2).multiply_vector(origin)

    print(f"  joint 1 (shoulder): angle={theta1} deg, link length={length1}")
    print(f"  joint 2 (elbow, relative): angle={theta2} deg, link length={length2}")
    print(f"  elbow position (x, y)        = ({elbow_position[0]:.4f}, {elbow_position[1]:.4f})")
    print(f"  end-effector position (x, y) = ({end_effector[0]:.4f}, {end_effector[1]:.4f})")

    # sanity check against the closed-form 2R planar arm equations
    t1, t2 = math.radians(theta1), math.radians(theta2)
    expected_x = length1 * math.cos(t1) + length2 * math.cos(t1 + t2)
    expected_y = length1 * math.sin(t1) + length2 * math.sin(t1 + t2)
    print(f"  closed-form check (x, y)     = ({expected_x:.4f}, {expected_y:.4f})")
    print()


# -- Challenge 4: homogeneous coordinates -------------------------------------

HOMOGENEOUS_COORDINATES_EXPLANATION = """
Why Engineers Augment Vectors with an Extra Coordinate
========================================================

The problem: translation is not a linear map.

A linear transformation is any map that can be written as y = A x for some
matrix A. Every linear map has one unavoidable property: it sends the
origin to the origin, because A * 0 = 0 for any matrix A. Rotation,
scaling, and shearing are all linear -- they can each be written as a
single matrix multiplying a vector, and Challenge 1 and 2 above show
exactly that.

Translation cannot be written that way. Shifting a point by (tx, ty) sends
the origin (0, 0) to (tx, ty), not to (0, 0). So no 2x2 matrix A satisfies
"A * v = v + t" for every v -- translation is affine (linear plus an
offset), not linear, and it sits outside the set of things ordinary
matrix multiplication can express.

The fix: add a constant extra coordinate.

Engineers augment an n-dimensional vector with one extra coordinate, most
commonly fixed at 1: (x, y) -> (x, y, 1). This is a "homogeneous
coordinate." In this augmented 3D space, translation becomes linear:

    [ 1  0  tx ]   [ x ]   [ x + tx ]
    [ 0  1  ty ] * [ y ] = [ y + ty ]
    [ 0  0  1  ]   [ 1 ]   [   1    ]

The extra row/column smuggles the additive offset into the matrix itself.
Rotation and scaling still work exactly as before -- they just get an
extra row and column of [0, 0, 1] padding, as in
homogeneous_rotation()/homogeneous_translation() above -- so a rotation,
a scale, and a translation can all be expressed as 3x3 matrices in the
*same* space, and composed by ordinary matrix multiplication:

    T_total = T3 @ T2 @ T1

instead of manually tracking "multiply by this matrix, then add this
offset" at every step. This is precisely what Challenge 3 relies on:
chaining several joints' rotate-then-translate transforms into one
combined matrix, then applying it once.

A second benefit: distinguishing points from directions.

The third coordinate doesn't have to be 1. Setting it to 0 instead --
(x, y, 0) -- represents a direction or displacement rather than a
location. Multiplying a (x, y, 0) vector by a translation matrix leaves it
unchanged (because tx and ty get multiplied by 0), which is the correct
physical behavior: translating the world shouldn't change which way an
arrow points, only where a point sits. This is why the robot arm's
end-effector was represented as (0, 0, 1) above -- it's a location -- while
a velocity or heading vector would use a trailing 0 instead.

In short: homogeneous coordinates aren't a trick for its own sake -- they
enlarge the space just enough that translation, rotation, and scaling all
become expressible as matrix multiplication, so an entire chain of
transformations (a robot arm, a camera projection, a graphics pipeline)
can be reduced to a single matrix multiply.
"""


def challenge_4():
    print("Challenge 4: homogeneous coordinates")
    print(HOMOGENEOUS_COORDINATES_EXPLANATION)

    # A quick demonstration of the core claim: no 2x2 matrix can translate.
    translate_by = (5, -3)
    origin_2d = [0, 0]
    best_effort = Matrix.identity(2).multiply_vector(origin_2d)  # any 2x2 A * [0,0] = [0,0]
    print(f"  any 2x2 matrix times the origin: {best_effort}  (always [0, 0] -- can't shift it)")

    translation_matrix = homogeneous_translation(*translate_by)
    origin_homogeneous = [0, 0, 1]
    shifted = translation_matrix.multiply_vector(origin_homogeneous)
    print(f"  homogeneous translation matrix times [0, 0, 1]: {shifted}  (successfully shifted)")
    print()


if __name__ == "__main__":
    challenge_1()
    challenge_2()
    challenge_3()
    challenge_4()
