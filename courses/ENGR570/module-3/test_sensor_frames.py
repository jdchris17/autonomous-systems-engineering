"""Tests for sensor_frames.py. Run with:  python -m pytest -v"""

import numpy as np
import pytest

from rotation_engine import (
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
    rotation_vector_to_matrix,
)
from sensor_frames import (
    FrameSet,
    PinholeCamera,
    Transform,
    attitude_to_transform,
    camera_attitude_from_boresight,
    celestial_to_earth_fixed,
    earth_from_celestial,
    radec_to_unit,
)

TOL = 1e-13

# Known orientation matrices used throughout.
C_CB = Transform("C", "B", rotation_vector_to_matrix([0.2, -0.5, 1.1]))
C_BG = Transform("B", "G", rotation_vector_to_matrix([-1.3, 0.4, 0.7]))
C_IB = Transform("I", "B", rotation_vector_to_matrix([0.01, 0.02, -0.03]))
STAR_G = radec_to_unit(120.0, 30.0)


def full_frame_set():
    frames = FrameSet()
    frames.add(C_CB)
    frames.add(C_BG)
    frames.add(C_IB)
    frames.add(earth_from_celestial(1.234))
    return frames


# --------------------------------------------------- requirement 1: labelling
def test_transform_knows_its_source_and_destination():
    assert (C_CB.to_frame, C_CB.from_frame) == ("C", "B")
    assert C_CB.name == "C_CB"
    assert "instrument body -> camera" in repr(C_CB)


def test_unknown_frame_is_rejected():
    with pytest.raises(ValueError, match="unknown frame"):
        Transform("C", "Q", np.eye(3))


def test_apply_converts_coordinates_from_source_to_destination():
    # Body frame = camera frame turned +90 deg about z, so the body x axis
    # is the camera y axis: columns of C_CB are the body axes in C.
    c_cb = Transform("C", "B", rotation_vector_to_matrix([0, 0, np.pi / 2]))
    np.testing.assert_allclose(c_cb.apply([1, 0, 0]), [0, 1, 0], atol=TOL)


# ------------------------------------------------- requirement 2: composition
def test_composition_matches_the_matrix_product():
    c_cg = C_CB @ C_BG                                  # C_CG = C_CB C_BG
    assert (c_cg.to_frame, c_cg.from_frame) == ("C", "G")
    np.testing.assert_allclose(c_cg.matrix, C_CB.matrix @ C_BG.matrix)
    np.testing.assert_allclose(c_cg.apply(STAR_G),
                               C_CB.apply(C_BG.apply(STAR_G)), atol=TOL)


def test_composition_in_the_wrong_order_is_refused():
    with pytest.raises(ValueError, match="cannot compose C_BG @ C_CB"):
        C_BG @ C_CB
    with pytest.raises(ValueError):
        C_CB @ C_CB


# ------------------------------------------------------ requirement 3: inverse
def test_inverse_is_the_transpose_with_swapped_frames():
    c_bc = C_CB.inverse()
    assert (c_bc.to_frame, c_bc.from_frame) == ("B", "C")
    np.testing.assert_array_equal(c_bc.matrix, C_CB.matrix.T)
    np.testing.assert_allclose((c_bc @ C_CB).matrix, np.eye(3), atol=TOL)


def test_inverse_of_a_product_reverses_the_chain():
    left = (C_CB @ C_BG).inverse()
    right = C_BG.inverse() @ C_CB.inverse()
    assert (left.to_frame, left.from_frame) == ("G", "C")
    np.testing.assert_allclose(left.matrix, right.matrix, atol=TOL)


# --------------------------------------------------- requirement 4: validation
def test_non_orthogonal_matrix_is_rejected():
    skewed = np.eye(3)
    skewed[0, 1] = 0.01
    with pytest.raises(ValueError, match="orthogonal"):
        Transform("C", "B", skewed)


def test_reflection_is_rejected():
    with pytest.raises(ValueError, match="proper"):
        Transform("C", "B", np.diag([1.0, -1.0, 1.0]))


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError):
        Transform("C", "B", np.eye(4))


def test_stored_matrix_is_read_only_and_products_stay_valid():
    with pytest.raises(ValueError):
        C_CB.matrix[0, 0] = 5.0
    chained = C_CB
    for _ in range(1000):                   # 1000 products, still a rotation
        chained = chained @ C_CB.inverse() @ C_CB
    np.testing.assert_allclose(chained.matrix.T @ chained.matrix, np.eye(3),
                               atol=1e-12)


# ------------------------------------- requirement 5: GCRS <-> camera and back
def test_star_round_trip_celestial_to_camera_and_back():
    c_cg = C_CB @ C_BG
    c_gc = c_cg.inverse()
    star_c = c_cg.apply(STAR_G)
    assert np.linalg.norm(star_c) == pytest.approx(1.0)
    np.testing.assert_allclose(c_gc.apply(star_c), STAR_G, atol=TOL)  # C_GC C_CG s_G = s_G
    np.testing.assert_allclose((c_gc @ c_cg).apply(STAR_G), STAR_G, atol=TOL)


def test_round_trip_for_a_whole_star_field():
    rng = np.random.default_rng(0)
    stars = rng.normal(size=(50, 3))
    stars /= np.linalg.norm(stars, axis=1, keepdims=True)
    c_cg = C_CB @ C_BG
    back = c_cg.inverse().apply(c_cg.apply(stars))
    np.testing.assert_allclose(back, stars, atol=TOL)


# ------------------------------------------------------------------ FrameSet
def test_frame_set_composes_stored_dcms():
    frames = full_frame_set()
    c_cg = frames.get("C", "G")
    assert c_cg.name == "C_CG"
    np.testing.assert_allclose(c_cg.matrix, C_CB.matrix @ C_BG.matrix, atol=TOL)
    np.testing.assert_allclose(frames.get("G", "C").matrix, c_cg.matrix.T,
                               atol=TOL)


def test_frame_set_longest_chain_camera_to_earth_and_imu():
    frames = full_frame_set()
    c_ec = frames.get("E", "C")             # C -> B -> G -> E
    expected = (earth_from_celestial(1.234).matrix @ C_BG.matrix.T
                @ C_CB.matrix.T)
    np.testing.assert_allclose(c_ec.matrix, expected, atol=TOL)
    c_ic = frames.get("I", "C")             # C -> B -> I
    np.testing.assert_allclose(c_ic.matrix, C_IB.matrix @ C_CB.matrix.T,
                               atol=TOL)


def test_frame_set_all_pairs_are_mutually_consistent():
    frames = full_frame_set()
    for a in "CBIGE":
        for b in "CBIGE":
            ab = frames.get(a, b)
            assert (ab.to_frame, ab.from_frame) == (a, b)
            np.testing.assert_allclose(ab.matrix, frames.get(b, a).matrix.T,
                                       atol=TOL)
            for c in "CBIGE":
                np.testing.assert_allclose(
                    (frames.get(a, b) @ frames.get(b, c)).matrix,
                    frames.get(a, c).matrix, atol=TOL)


def test_frame_set_same_frame_is_identity():
    np.testing.assert_array_equal(FrameSet().get("C", "C").matrix, np.eye(3))


def test_frame_set_refuses_duplicates_and_loops():
    frames = full_frame_set()
    with pytest.raises(ValueError, match="already defined"):
        frames.add(C_CB)
    with pytest.raises(ValueError, match="already defined"):
        frames.add(C_CB.inverse())
    with pytest.raises(ValueError, match="already connected"):
        frames.add(Transform("C", "G", np.eye(3)))


def test_frame_set_replace_updates_both_directions():
    frames = full_frame_set()
    new = Transform("C", "B", np.eye(3))
    frames.add(new, replace=True)
    np.testing.assert_array_equal(frames.get("B", "C").matrix, np.eye(3))


def test_frame_set_missing_chain_raises():
    frames = FrameSet()
    frames.add(C_CB)
    with pytest.raises(KeyError, match="no known transform chain"):
        frames.get("C", "G")
    with pytest.raises(ValueError, match="unknown frame"):
        frames.get("C", "Z")


def test_frame_set_transform_vectors():
    frames = full_frame_set()
    star_c = frames.transform(STAR_G, "C", "G")
    np.testing.assert_allclose(frames.transform(star_c, "G", "C"), STAR_G,
                               atol=TOL)


# ------------------------------------------------------ Module 2 integration
@pytest.mark.parametrize("angle", [0.0, 0.5, 2.0, 4.019536973826625, -1.0])
def test_earth_frame_reuses_module_2_rotation(angle):
    c_eg = earth_from_celestial(angle)
    assert (c_eg.to_frame, c_eg.from_frame) == ("E", "G")
    np.testing.assert_allclose(c_eg.apply(STAR_G),
                               celestial_to_earth_fixed(STAR_G, angle),
                               atol=1e-15)


def test_star_at_ra_equal_to_earth_angle_is_on_greenwich_meridian():
    c_eg = earth_from_celestial(np.radians(120.0))
    np.testing.assert_allclose(c_eg.apply(radec_to_unit(120.0, 0.0)),
                               [1, 0, 0], atol=TOL)


# ------------------------------------------------------- attitude quaternions
def test_attitude_quaternion_columns_are_camera_axes():
    q = camera_attitude_from_boresight(120.0, 30.0)
    c_gc = attitude_to_transform(q)
    assert c_gc.name == "C_GC"
    np.testing.assert_allclose(c_gc.apply([0, 0, 1]), STAR_G, atol=TOL)
    np.testing.assert_allclose(c_gc.inverse().apply(STAR_G), [0, 0, 1],
                               atol=TOL)


def test_zero_roll_puts_north_up_and_east_left():
    c_cg = attitude_to_transform(
        camera_attitude_from_boresight(120.0, 30.0)).inverse()
    north_star = c_cg.apply(radec_to_unit(120.0, 31.0))   # 1 deg north
    east_star = c_cg.apply(radec_to_unit(121.0, 30.0))    # slightly east
    assert north_star[1] < 0 and abs(north_star[0]) < 1e-12   # up: -y
    assert east_star[0] < 0                                    # left: -x


def test_roll_rotates_about_the_boresight():
    q0 = camera_attitude_from_boresight(40.0, -20.0, roll_deg=0.0)
    q90 = camera_attitude_from_boresight(40.0, -20.0, roll_deg=90.0)
    expected = quaternion_multiply(q0, axis_angle_to_quaternion([0, 0, 1],
                                                                np.pi / 2))
    np.testing.assert_allclose(quaternion_to_matrix(q90),
                               quaternion_to_matrix(expected), atol=TOL)
    # Boresight is unchanged by roll.
    np.testing.assert_allclose(quaternion_to_matrix(q90)[:, 2],
                               radec_to_unit(40.0, -20.0), atol=TOL)


# ---------------------------------------------------------------- camera model
def test_nominal_focal_length():
    cam = PinholeCamera()
    assert cam.fx == pytest.approx(1920 / (2 * np.tan(np.radians(10.0))))
    assert cam.fx == pytest.approx(5444.43, abs=0.01)
    assert cam.fy == cam.fx
    assert cam.vertical_fov_deg == pytest.approx(11.33, abs=0.01)


def test_boresight_projects_to_principal_point():
    pixels, visible = PinholeCamera().project([0, 0, 1])
    np.testing.assert_allclose(pixels, [[960.0, 540.0]])
    assert visible[0]


def test_edges_of_the_field_of_view():
    cam = PinholeCamera()
    half = np.radians(10.0)
    pixels, _ = cam.project([[np.sin(half), 0, np.cos(half)],
                             [-np.sin(half), 0, np.cos(half)]])
    np.testing.assert_allclose(pixels[:, 0], [1920.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(pixels[:, 1], [540.0, 540.0])


def test_axes_map_to_right_and_down():
    cam = PinholeCamera()
    pixels, _ = cam.project([[0.01, 0, 1], [0, 0.01, 1]])
    assert pixels[0, 0] > 960 and pixels[0, 1] == pytest.approx(540)
    assert pixels[1, 1] > 540 and pixels[1, 0] == pytest.approx(960)


def test_visibility_rejects_behind_and_off_sensor():
    cam = PinholeCamera()
    pixels, visible = cam.project([
        [0, 0, 1],                 # centre
        [0, 0, -1],                # directly behind
        [0.0, 0.0, 0.0 + 1e-300],  # in front, at centre
        [1, 0, 0],                 # 90 deg off axis
        [0.3, 0, 1],               # in front but beyond the right edge
        [0, 0.15, 1],              # below the bottom edge
        [-0.05, 0.05, -1],         # behind: would alias onto the sensor
    ])
    assert list(visible) == [True, False, True, False, False, False, False]
    assert np.all(np.isnan(pixels[[1, 3, 6]]))


def test_projection_ignores_vector_length():
    cam = PinholeCamera()
    a, _ = cam.project([0.02, -0.03, 1.0])
    b, _ = cam.project([0.2, -0.3, 10.0])
    np.testing.assert_allclose(a, b)


def test_unproject_inverts_project():
    cam = PinholeCamera()
    rng = np.random.default_rng(1)
    rays = np.column_stack((rng.uniform(-0.17, 0.17, 100),
                            rng.uniform(-0.09, 0.09, 100), np.ones(100)))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    pixels, visible = cam.project(rays)
    assert visible.all()
    np.testing.assert_allclose(cam.unproject(pixels), rays, atol=1e-14)
