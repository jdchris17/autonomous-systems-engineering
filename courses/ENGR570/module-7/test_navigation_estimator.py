"""Tests for navigation_estimator.py. Run with:  python -m pytest -v"""

import numpy as np
import pytest

import navigation_estimator as ne
import navigation_scenarios as ns
from gravity_estimator import tangent_basis

cp = ne.cp
ARCSEC = ne.ARCSEC
C_EG, _ = cp.earth_orientation(ns.START_UTC)
C_GE = np.asarray(C_EG.matrix).T
TRUE_STATE = np.radians(ns.TRUE_POSITION)


def synthetic_measurements(n, sigma_arcsec=5.0, seed=0, state=TRUE_STATE,
                           noise=True, sigmas=None):
    """Measurements built directly from the model, without the sensor chain."""
    rng = np.random.default_rng(seed)
    truth = ne.predict_celestial_vertical(state, C_GE)
    basis = tangent_basis(truth)
    out = []
    for k in range(n):
        sigma = (sigma_arcsec if sigmas is None else sigmas[k]) * ARCSEC
        z = truth + (basis.T @ rng.normal(0, sigma, 2) if noise else 0)
        z = z / np.linalg.norm(z)
        b = tangent_basis(z)
        out.append(ne.VerticalMeasurement(
            up_celestial=z, covariance=sigma**2 * b.T @ b, C_GE=C_GE,
            utc=ns.START_UTC))
    return out


# ------------------------------------------------------------ model + Jacobian
def test_prediction_matches_module_6_geometry():
    predicted = ne.predict_celestial_vertical(TRUE_STATE, C_GE)
    np.testing.assert_allclose(
        predicted, C_GE @ ns.ps.true_vertical_ecef(*ns.TRUE_POSITION), atol=1e-15)
    lat, lon = cp.celestial_vertical_to_position(predicted, C_GE.T)
    assert (lat, lon) == pytest.approx(ns.TRUE_POSITION, abs=1e-11)


@pytest.mark.parametrize("lat, lon", [(45.0, -75.0), (0.0, 0.0), (-60.0, 170.0),
                                      (85.0, 10.0)])
def test_analytic_jacobian_matches_finite_differences(lat, lon):
    state = np.radians([lat, lon])
    J = ne.celestial_vertical_jacobian(state, C_GE)
    assert J.shape == (3, 2)
    eps = 1e-7
    for i in range(2):
        step = np.zeros(2)
        step[i] = eps
        numeric = (ne.predict_celestial_vertical(state + step, C_GE)
                   - ne.predict_celestial_vertical(state - step, C_GE)) / (2 * eps)
        np.testing.assert_allclose(J[:, i], numeric, atol=1e-9)
    # Columns are tangent to the sphere, with lengths 1 and cos(lat).
    up = ne.predict_celestial_vertical(state, C_GE)
    np.testing.assert_allclose(J.T @ up, 0, atol=1e-15)
    assert np.linalg.norm(J[:, 0]) == pytest.approx(1.0)
    assert np.linalg.norm(J[:, 1]) == pytest.approx(np.cos(state[0]))


def test_radians_to_metres_uses_wgs84_radii():
    T = ne.radians_to_metres(0.0)
    assert np.radians(1.0) * T[0, 0] == pytest.approx(110574.3, abs=0.5)
    assert np.radians(1.0) * T[1, 1] == pytest.approx(111319.5, abs=0.5)
    assert ne.radians_to_metres(np.radians(60.0))[1, 1] == pytest.approx(
        0.5 * 6394209.2, rel=1e-6)


# ----------------------------------------------------------------- estimation
def test_noise_free_batch_recovers_the_position_from_a_poor_start():
    result = ne.batch_position_estimate(synthetic_measurements(5, noise=False),
                                        (30.0, -40.0))
    assert result.converged
    assert cp.horizontal_position_error(result.lat_lon, ns.TRUE_POSITION) < 1e-5
    assert result.chi2 < 1e-12
    assert result.dof == 8


def test_covariance_matches_the_analytic_value_and_scales_with_n():
    sigma = 5.0
    expected_m = sigma * ARCSEC * 6.37e6
    for n in (1, 4, 25):
        result = ne.batch_position_estimate(synthetic_measurements(n, sigma),
                                            ns.TRUE_POSITION)
        sigmas = np.sqrt(np.diag(result.position_covariance_ne))
        np.testing.assert_allclose(sigmas, expected_m / np.sqrt(n), rtol=0.01)
        assert abs(result.position_covariance_ne[0, 1]) < 1e-6 * sigmas.prod()


def test_single_measurement_reproduces_the_direct_inversion():
    m = synthetic_measurements(1, seed=3)
    result = ne.batch_position_estimate(m, (44.0, -74.0))
    direct = ns.single_fix(m[0])
    assert cp.horizontal_position_error(result.lat_lon, direct) < 1e-4
    assert result.dof == 0 and np.isnan(result.reduced_chi2)


def test_residuals_are_whitened():
    # Whitened residuals of correctly modelled noise have unit variance,
    # whatever the measurement sigmas are.
    sigmas = np.tile([1.0, 50.0], 200)
    result = ne.batch_position_estimate(
        synthetic_measurements(400, sigmas=sigmas, seed=4), ns.TRUE_POSITION)
    assert result.reduced_chi2 == pytest.approx(1.0, abs=0.12)
    pairs = result.residuals.reshape(-1, 2)
    assert np.std(pairs[0::2]) == pytest.approx(1.0, abs=0.15)
    assert np.std(pairs[1::2]) == pytest.approx(1.0, abs=0.15)


def test_weighting_reduces_the_influence_of_noisy_measurements():
    sigmas = np.array([5.0] * 15 + [50.0] * 5)
    result = ne.batch_position_estimate(
        synthetic_measurements(20, sigmas=sigmas, seed=5), ns.TRUE_POSITION)
    fractions = result.diagnostics["information_fraction"]
    assert fractions.sum() == pytest.approx(1.0)
    assert fractions[15:].sum() == pytest.approx(5 / (15 * 100 + 5), rel=0.02)
    # Five noisy points add almost nothing to fifteen good ones.
    good_only = ne.batch_position_estimate(
        synthetic_measurements(15, sigma_arcsec=5.0, seed=5), ns.TRUE_POSITION)
    ratio = (np.trace(result.position_covariance_ne)
             / np.trace(good_only.position_covariance_ne))
    assert ratio == pytest.approx(1500 / 1505, rel=0.01)


def test_weighted_batch_is_statistically_consistent():
    values = []
    sigmas = np.array([5.0] * 15 + [50.0] * 5)
    for seed in range(400):
        result = ne.batch_position_estimate(
            synthetic_measurements(20, sigmas=sigmas, seed=100 + seed),
            ns.TRUE_POSITION)
        error = ns.north_east_error(result.lat_lon)
        values.append(ne.nees(error, result.position_covariance_ne))
    assert np.mean(values) == pytest.approx(2.0, abs=0.3)
    assert np.mean(np.array(values) < 5.991) == pytest.approx(0.95, abs=0.04)


def test_input_validation():
    with pytest.raises(ValueError, match="at least one"):
        ne.batch_position_estimate([], ns.TRUE_POSITION)
    with pytest.raises(ValueError, match="clock_model"):
        ne.batch_position_estimate(synthetic_measurements(3), ns.TRUE_POSITION,
                                   clock_model="banana")
    with pytest.raises(ValueError, match="unknown shared error"):
        ne.batch_position_estimate(synthetic_measurements(3), ns.TRUE_POSITION,
                                   shared_errors={"gremlins": np.eye(3)})
    with pytest.raises(ValueError, match="C_GC"):
        ne.batch_position_estimate(synthetic_measurements(3), ns.TRUE_POSITION,
                                   estimate_calibration=True)
    bad = synthetic_measurements(3)
    bad[1].covariance = np.zeros((3, 3))
    with pytest.raises(ValueError, match="positive definite"):
        ne.batch_position_estimate(bad, ns.TRUE_POSITION)


# ------------------------------------------------- full sensor chain (Module 6)
def test_sensor_chain_measurement_and_covariance():
    measurements, truth = ns.simulate_batch(np.random.default_rng(1), 3)
    m = measurements[0]
    true_up = truth["observations"][0]["up_celestial"]
    assert np.degrees(np.arccos(np.clip(m.up_celestial @ true_up, -1, 1))) < 0.02
    assert np.linalg.norm(m.up_celestial) == pytest.approx(1.0)
    # Rank two, no component along the vertical itself.
    np.testing.assert_allclose(m.covariance @ m.up_celestial, 0, atol=1e-18)
    assert np.linalg.matrix_rank(m.covariance, tol=1e-18) == 2
    np.testing.assert_allclose(m.C_GE, C_GE, atol=1e-15)
    assert "IERS" in m.earth_orientation_source
    assert m.quality.verdict == "steady"
    assert m.jacobian_accel.shape == m.jacobian_calibration.shape == (3, 3)


def test_measurement_covariance_matches_monte_carlo_scatter():
    rng = np.random.default_rng(2)
    measurements, truth = ns.simulate_batch(rng, 600, spacing_s=0.0,
                                            run_quality_check=False)
    true_up = truth["observations"][0]["up_celestial"]
    basis = tangent_basis(true_up)
    errors = np.array([basis @ (m.up_celestial - true_up) for m in measurements])
    predicted = np.mean([basis @ m.covariance @ basis.T for m in measurements],
                        axis=0)
    np.testing.assert_allclose(np.cov(errors.T), predicted,
                               rtol=0.2, atol=0.1 * np.trace(predicted))


def test_batch_from_sensors_is_accurate_and_consistent():
    values, errors = [], []
    for seed in range(60):
        measurements, _ = ns.simulate_batch(np.random.default_rng(10 + seed),
                                            run_quality_check=False)
        result = ne.batch_position_estimate(measurements,
                                            ns.single_fix(measurements[0]))
        error = ns.north_east_error(result.lat_lon)
        errors.append(np.linalg.norm(error))
        values.append(ne.nees(error, result.position_covariance_ne))
    assert np.sqrt(np.mean(np.square(errors))) < 150.0       # metres
    assert 1.3 < np.mean(values) < 2.9


# ----------------------------------------------------------------- clock state
def test_simple_clock_model_is_rank_deficient_and_covariance_is_withheld():
    measurements = synthetic_measurements(10, seed=6)
    result = ne.batch_position_estimate(measurements, ns.TRUE_POSITION,
                                        estimate_clock=True)
    assert result.state_names == ["latitude", "longitude", "clock"]
    assert result.observability == "rank deficient"
    assert result.covariance is None and result.position_covariance_ne is None
    assert result.singular_values[-1] < 1e-8 * result.singular_values[0]
    # The null direction trades longitude against clock at the sidereal rate.
    J = result.jacobian
    null = np.array([0.0, ne.EARTH_ROTATION_RATE_UT1, -1.0])
    np.testing.assert_allclose(J @ null, 0, atol=1e-9 * np.abs(J).max())


def test_clock_jacobian_column_matches_finite_differences():
    measurements = synthetic_measurements(3, seed=7)
    problem = ne._Problem(measurements, True, False, "simple", None, None)
    x = np.array([*TRUE_STATE, 2.0])
    analytic = problem.raw_jacobian(x)
    eps = 1e-3
    numeric = np.empty_like(analytic)
    for i in range(3):
        step = np.zeros(3)
        step[i] = eps if i == 2 else 1e-7
        hi = np.einsum("kij,kj->ki", problem.basis, problem.predict(x + step)).ravel()
        lo = np.einsum("kij,kj->ki", problem.basis, problem.predict(x - step)).ravel()
        numeric[:, i] = (hi - lo) / (2 * step[i])
    np.testing.assert_allclose(analytic, numeric, atol=1e-8)


def test_longitude_absorbs_an_unmodelled_clock_error():
    measurements, _ = ns.simulate_batch(np.random.default_rng(3), 5,
                                        clock_offset_s=10.0,
                                        run_quality_check=False)
    result = ne.batch_position_estimate(measurements, ns.single_fix(measurements[0]))
    shift = cp.longitude_difference(result.longitude_deg, -75.0)
    assert shift == pytest.approx(
        -np.degrees(ne.EARTH_ROTATION_RATE_UT1 * 10.0), abs=0.005)
    assert ne.assess_batch_result(result, measurements).status == "VALID"


# ----------------------------------------------------------- calibration states
def calibration_batch(attitudes, seed, misalignment):
    return ns.simulate_batch(np.random.default_rng(seed), len(attitudes),
                             attitudes=attitudes, misalignment_rad=misalignment,
                             run_quality_check=False)


def test_calibration_jacobian_matches_finite_differences():
    measurements, _ = calibration_batch(ns.diverse_attitudes(4), 4, None)
    problem = ne._Problem(measurements, False, True, "simple", None, None)
    x = np.array([*TRUE_STATE, 2e-4, -1e-4, 3e-4])
    analytic = problem.raw_jacobian(x)
    numeric = np.empty_like(analytic)
    for i in range(5):
        step = np.zeros(5)
        step[i] = 1e-7
        hi = np.einsum("kij,kj->ki", problem.basis, problem.predict(x + step)).ravel()
        lo = np.einsum("kij,kj->ki", problem.basis, problem.predict(x - step)).ravel()
        numeric[:, i] = (hi - lo) / 2e-7
    np.testing.assert_allclose(analytic, numeric, atol=5e-4)


def test_calibration_is_unobservable_from_a_fixed_attitude():
    misalignment = np.radians([0.05, -0.03, 0.04])
    measurements, _ = calibration_batch([ns.DEFAULT_ATTITUDE] * 20, 5, misalignment)
    result = ne.batch_position_estimate(measurements, ns.single_fix(measurements[0]),
                                        estimate_calibration=True)
    assert result.observability != "well conditioned"
    assert result.covariance is None


def test_calibration_and_position_are_recovered_from_diverse_attitudes():
    misalignment = np.radians([0.05, -0.03, 0.04])
    measurements, _ = calibration_batch(ns.diverse_attitudes(20), 6, misalignment)
    start = ns.single_fix(measurements[0])
    ignoring = ne.batch_position_estimate(measurements, start)
    solving = ne.batch_position_estimate(measurements, start,
                                         estimate_calibration=True)
    assert solving.observability == "well conditioned"
    sigma = np.sqrt(np.diag(solving.covariance))[2:]
    np.testing.assert_allclose(solving.calibration_rad, misalignment,
                               atol=5 * sigma.max())
    assert np.all(sigma < 30 * ARCSEC)
    error_solving = cp.horizontal_position_error(solving.lat_lon, ns.TRUE_POSITION)
    assert error_solving < 1000.0
    assert ne.nees(ns.north_east_error(solving.lat_lon),
                   solving.position_covariance_ne) < 20
    # The unmodelled misalignment makes the plain fit visibly inconsistent.
    assert ignoring.reduced_chi2 > 20 * solving.reduced_chi2


# ----------------------------------------------------------------- shared errors
def test_shared_error_covariance_does_not_average_down():
    P_b = ns.ACCEL_BIAS_SIGMA**2 * np.eye(3)
    P_cal = ns.MISALIGNMENT_SIGMA**2 * np.eye(3)
    traces = {}
    for n in (5, 20, 80):
        measurements, _ = ns.simulate_batch(np.random.default_rng(7), n,
                                            run_quality_check=False)
        start = ns.single_fix(measurements[0])
        naive = ne.batch_position_estimate(measurements, start)
        aware = ne.batch_position_estimate(
            measurements, start,
            shared_errors={"accelerometer_bias": P_b, "calibration": P_cal})
        traces[n] = (np.trace(naive.position_covariance_ne),
                     np.trace(aware.position_covariance_ne))
        # Same data, nearly the same answer (the correlated-error weighting
        # shifts it by a few random sigmas at most); what changes is the
        # stated confidence, by two orders of magnitude.
        assert cp.horizontal_position_error(naive.lat_lon, aware.lat_lon) < 500.0
    assert traces[5][0] / traces[80][0] == pytest.approx(16.0, rel=0.2)
    assert traces[5][1] / traces[80][1] == pytest.approx(1.0, abs=0.01)
    assert np.sqrt(traces[80][1]) > 5000.0            # kilometres, not metres


# ----------------------------------------------------------- acceptance policy
def assess(measurements, **kwargs):
    result = ne.batch_position_estimate(measurements, ns.single_fix(measurements[0]),
                                        **kwargs)
    return result, ne.assess_batch_result(result, measurements)


def test_clean_solution_is_valid_with_the_standing_caveat():
    measurements, _ = ns.simulate_batch(np.random.default_rng(8))
    _, assessment = assess(measurements)
    assert assessment.status == "VALID"
    assert any("assumed, not verified" in r for r in assessment.reasons)
    assert "cannot verify" in assessment.caveat
    assert assessment.metrics["observability"] == "well conditioned"
    assert 0.3 < assessment.metrics["reduced_chi2"] < 2.0


def test_anomalous_measurement_is_rejected_and_identified():
    measurements, _ = ns.simulate_batch(np.random.default_rng(9))
    bad = measurements[11]
    kick = tangent_basis(bad.up_celestial)[0] * 300 * ARCSEC
    bad.up_celestial = (bad.up_celestial + kick) / np.linalg.norm(bad.up_celestial + kick)
    _, assessment = assess(measurements)
    assert assessment.status == "REJECTED"
    assert any("[11]" in r for r in assessment.reasons)
    del measurements[11]
    _, clean = assess(measurements)
    assert clean.status == "VALID"


def test_unobservable_clock_state_is_reported():
    measurements, _ = ns.simulate_batch(np.random.default_rng(10), 6)
    result, assessment = assess(measurements, estimate_clock=True)
    assert assessment.status == "UNOBSERVABLE"
    assert any("longitude / clock" in r for r in assessment.reasons)
    assert result.covariance is None
    assert any("no covariance" in r for r in assessment.reasons)


def test_uncertain_earth_orientation_degrades_the_solution():
    measurements, _ = ns.simulate_batch(np.random.default_rng(11))
    result = ne.batch_position_estimate(measurements, ns.single_fix(measurements[0]))
    stale = {"available": False, "source": "none", "is_prediction": False,
             "ut1_sigma_s": 0.9}
    assessment = ne.assess_batch_result(result, measurements, stale)
    assert assessment.status == "DEGRADED"
    assert assessment.metrics["reference_east_error_m"] == pytest.approx(296, abs=5)
    assert any("Earth-orientation" in r for r in assessment.reasons)


def test_disturbed_accelerometer_is_rejected():
    measurements, _ = ns.simulate_batch(np.random.default_rng(12), 5)

    class Flag:
        detectable_disturbance = True
        verdict = "disturbed"

    measurements[2].quality = Flag()
    _, assessment = assess(measurements)
    assert assessment.status == "REJECTED"
    assert any("measurement 2" in r for r in assessment.reasons)


def test_accelerating_instrument_passes_every_check_and_is_wrong():
    measurements, _ = ns.simulate_batch(np.random.default_rng(13),
                                        acceleration_enu=(0.05, 0.0, 0.0))
    result, assessment = assess(measurements)
    error = cp.horizontal_position_error(result.lat_lon, ns.TRUE_POSITION)
    assert error > 30_000.0
    assert assessment.status == "VALID"                 # self-consistent
    assert error > 100 * np.sqrt(np.trace(result.position_covariance_ne))
    assert "constant acceleration" in assessment.caveat


def test_non_convergence_and_missing_redundancy():
    measurements = synthetic_measurements(4, seed=14)
    result = ne.batch_position_estimate(measurements, ns.TRUE_POSITION)
    failed = ne.assess_navigation_solution(
        result.state, result.covariance, result.jacobian, result.residuals,
        [None] * 4, None, converged=False)
    assert failed.status == "REJECTED"
    one = ne.batch_position_estimate(measurements[:1], ns.TRUE_POSITION)
    lonely = ne.assess_navigation_solution(
        one.state, one.covariance, one.jacobian, one.residuals, [None], None)
    assert lonely.status == "DEGRADED"
    assert any("zero degrees of freedom" in r for r in lonely.reasons)


def test_reference_status_for_the_observation_time():
    status = ne.reference_status(ns.START_UTC)
    assert status["available"] and status["is_prediction"]
    assert 0 < status["days_predicted"] < 400
    assert 0 < status["ut1_sigma_s"] < 0.05
    assert not ne.reference_status("2060-01-01T00:00:00")["available"]


def test_observability_classification():
    assert ne.classify_observability([2.0, 1.0]) == "well conditioned"
    assert ne.classify_observability([2.0, 1e-5]) == "poorly conditioned"
    assert ne.classify_observability([2.0, 1e-12]) == "rank deficient"
    assert ne.classify_observability([]) == "rank deficient"
