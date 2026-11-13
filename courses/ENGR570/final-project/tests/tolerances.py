"""
Every tolerance and seed used by the system verification suites, with the
reason for its value. The verification report prints this table.
"""

import os

SEED = 5709                      # base seed; each test adds a fixed offset
MONTE_CARLO_TRIALS = int(os.environ.get("ENDURIA_TRIALS", "500"))

TOLERANCES = {
    "ideal_position_m": (
        1e-3,
        "Perfect, noise-free, correctly modelled case. 1 mm is far below any "
        "operational significance and far above double-precision rounding "
        "(about 1e-9 m at Earth radius). Timestamps carry nanoseconds."),
    "independent_position_m": (
        0.01,
        "Engine versus an Astropy-generated observation. Measured agreement is "
        "2 to 8 micrometres, because both chains end in the same ERFA "
        "routines. 1 cm is 0.3 milliarcseconds of direction: any error in "
        "frames, signs, time scales or iteration would exceed it by orders of "
        "magnitude."),
    "independent_direction_mas": (
        0.1,
        "Star directions, engine chain versus Astropy frames, with and without "
        "atmosphere. 0.1 mas is 3 mm on the ground."),
    "earth_rotation_rad": (
        1e-11,
        "GCRS to ITRS rotation, engine versus Module 2 and Astropy: same IAU "
        "2006/2000A model; 1e-11 rad is 0.06 mm at the surface."),
    "geodesy_m": (
        1e-6,
        "Closed-form ellipsoid geometry against Astropy EarthLocation."),
    "distance_relative": (
        1e-9,
        "GeographicLib (Karney) against an independent Vincenty iteration, "
        "which itself converges to about 0.1 mm."),
    "calibration_arcsec": (
        0.5,
        "Recovered camera-IMU rotation in the noise-free diverse-attitude "
        "case; limited by the declared noise floor and the prior."),
    "coverage_sigmas": (
        3.5,
        "Monte Carlo coverage of the 95 % ellipse must lie within this many "
        "binomial standard errors of 95 %. With 500 trials one standard error "
        "is 0.97 percentage points."),
    "nees_sigmas": (
        3.5,
        "Mean NEES (2 degrees of freedom) must lie within this many standard "
        "errors, 2/sqrt(trials), of 2."),
}


def tolerance(name):
    return TOLERANCES[name][0]
