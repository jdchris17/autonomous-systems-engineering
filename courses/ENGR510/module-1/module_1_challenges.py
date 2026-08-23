"""Worked examples applying Vector to physical problems.

Frame convention used throughout: x = East, y = North, z = Up (ENU).
"""

import math

from vector import Vector


# -- rotation helpers ---------------------------------------------------
# Not part of Vector's core API -- these are 3D rotation matrices
# (rotation about a single axis) applied by hand via their standard
# trig formulas, since Vector doesn't implement matrix multiplication.

def rotate_x(v: Vector, degrees: float) -> Vector:
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    return Vector(v.x, v.y * c - v.z * s, v.y * s + v.z * c)


def rotate_y(v: Vector, degrees: float) -> Vector:
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    return Vector(v.z * s + v.x * c, v.y, v.z * c - v.x * s)


def rotate_z(v: Vector, degrees: float) -> Vector:
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    return Vector(v.x * c - v.y * s, v.x * s + v.y * c, v.z)


# -- Challenge 1: aircraft velocity --------------------------------------
# Math: scalar multiplication. Speed (a scalar, 250 knots) times a unit
# direction vector (north) gives a velocity vector -- magnitude = speed,
# direction = heading.

def challenge_1():
    print("Challenge 1: aircraft flying north at 250 knots")
    north = Vector(0, 1, 0)  # unit vector, direction only
    speed = 250              # knots, a scalar magnitude
    velocity = north * speed
    print(f"  direction (unit vector) = {north}")
    print(f"  velocity vector         = {velocity}")
    print(f"  |velocity| (speed check) = {velocity.magnitude()} knots")
    print()


# -- Challenge 2: displacement between two GPS fixes ----------------------
# Math: vector subtraction. Displacement = position_B - position_A.
# The magnitude of that difference is distance traveled; normalizing it
# gives the heading of travel.
#
# Real lat/lon isn't a flat vector space, so positions here are already
# converted to a local flat-earth approximation in meters (ENU), which is
# the standard simplification for short-range GPS displacement.

def challenge_2():
    print("Challenge 2: displacement between two GPS positions")
    position_a = Vector(1200, 4300, 0)   # meters, local ENU coordinates
    position_b = Vector(1500, 5100, 50)  # meters, local ENU coordinates

    displacement = position_b - position_a  # vector subtraction
    distance = displacement.magnitude()      # magnitude of the difference
    heading = displacement.normalize()       # direction of travel, unit vector

    print(f"  position A     = {position_a}")
    print(f"  position B     = {position_b}")
    print(f"  displacement   = {displacement}  (B - A)")
    print(f"  distance       = {distance:.2f} m")
    print(f"  heading (unit) = {heading}")
    print()


# -- Challenge 3: gravity vector under a rotated frame ---------------------
# Math: rotation transformation (change of basis). Rotating the frame
# recomputes the vector's x/y/z *components* in the new axes, but the
# vector itself -- its magnitude and its real, physical direction -- is
# unchanged. This is the difference between a vector (frame-independent)
# and its coordinates (frame-dependent).

def challenge_3():
    print("Challenge 3: gravity vector, rotated coordinate frame")
    gravity = Vector(0, 0, -9.81)  # straight down, in a level (Up = +z) frame
    print(f"  gravity in level frame        = {gravity}")
    print(f"  |gravity|                     = {gravity.magnitude():.4f} m/s^2")

    # Tilt the frame 30 degrees about the x-axis (e.g. a banking aircraft's
    # body frame vs. the earth frame).
    tilted = rotate_x(gravity, 30)
    print(f"  gravity in frame rolled 30deg = {tilted}")
    print(f"  |gravity| after rotation      = {tilted.magnitude():.4f} m/s^2")
    print(
        "  -> components (x, y, z) changed, but magnitude stayed the same:\n"
        "     rotation only changes which axes you're measuring against,\n"
        "     not the physical vector gravity represents."
    )
    print()


# -- Challenge 4: star observation as a unit vector ------------------------
# Math: normalization (v / |v|). A camera measures a direction to a star,
# not a reliable distance -- stars are effectively at infinity and any
# distance estimate from a single 2D image is unusable. Normalizing
# discards the (meaningless) magnitude and keeps only the direction, so
# the observation can be safely compared/combined with other unit vectors
# (e.g. via dot product for angular separation) regardless of scale.

def challenge_4():
    print("Challenge 4: star observation as a unit vector")
    raw_observation = Vector(120.0, 340.0, 890.0)  # arbitrary camera-frame units
    line_of_sight = raw_observation.normalize()

    print(f"  raw observation vector = {raw_observation}")
    print(f"  normalized (unit) LOS  = {line_of_sight}")
    print(f"  |LOS|                  = {line_of_sight.magnitude():.4f}")
    print(
        "  -> normalizing throws away the (meaningless) magnitude and keeps\n"
        "     only direction, so two observations can be compared purely by\n"
        "     angle (via dot product) without distance skewing the result."
    )
    print()


if __name__ == "__main__":
    challenge_1()
    challenge_2()
    challenge_3()
    challenge_4()
