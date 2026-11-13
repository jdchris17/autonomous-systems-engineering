"""
references.py - locally available astronomical reference data.

ReferenceData holds the star catalog and the Earth-orientation parameters
(EOP) the engine may use, with their provenance and age. Nothing is
downloaded: EOP come from the IERS table bundled with Astropy.

All astronomy is done by ERFA (the open implementation of the IAU SOFA
standards): space motion (pmsafe), the ICRS-to-observed chain (atco13), and
the IAU 2006/2000A celestial-to-terrestrial rotation. This module only
arranges the calls so that one set of EOP values is used consistently for
the stars and for the Earth rotation, and so that a truth simulator can
evaluate the same chain with perturbed values.
"""

from dataclasses import dataclass, replace

import erfa
import numpy as np
from astropy.time import Time

from .astrometry import J2000_JD, MAS, VACUUM, CatalogStar, prediction_uncertainty
from .estimation import ARCSEC, reference_status
from .geodesy import local_axes_ecef
from .timekeeping import iers_status, polar_motion

ISO_PRECISION = 9                    # nanoseconds: 1 ms of time is 0.46 m of longitude
MEASURED_UT1_SIGMA_S = 2e-5
MEASURED_POLAR_SIGMA_ARCSEC = 5e-5
NO_EOP_UT1_SIGMA_S = 0.9             # |UT1 - UTC| < 0.9 s is all that is guaranteed
NO_EOP_POLAR_SIGMA_ARCSEC = 0.3      # typical size of the polar motion itself


def utc_time(utc):
    """Astropy Time in UTC that prints with nanosecond precision."""
    time = utc if isinstance(utc, Time) else Time(utc, scale="utc")
    time = time.utc
    time.precision = ISO_PRECISION
    return time


@dataclass(frozen=True)
class EarthOrientationRecord:
    """Earth-orientation values for one instant, with where they came from."""
    utc: str
    utc_jd: tuple
    tt_jd: tuple
    tdb_jd: tuple
    dut1_s: float                    # UT1 - UTC
    xp_rad: float
    yp_rad: float
    available: bool
    source: str
    is_prediction: bool
    days_predicted: float
    ut1_sigma_s: float
    polar_motion_sigma_arcsec: float

    def with_errors(self, dut1_error_s=0.0, xp_error_rad=0.0, yp_error_rad=0.0):
        """The same instant with different EOP values (for truth simulation)."""
        return replace(self, dut1_s=self.dut1_s + dut1_error_s,
                       xp_rad=self.xp_rad + xp_error_rad,
                       yp_rad=self.yp_rad + yp_error_rad)

    def earth_from_celestial(self):
        """C_EG: GCRS -> ITRS, IAU 2006/2000A, CIO based."""
        tt1, tt2 = self.tt_jd
        x, y, s = erfa.xys06a(tt1, tt2)
        ut11, ut12 = erfa.utcut1(self.utc_jd[0], self.utc_jd[1], self.dut1_s)
        polar = erfa.pom00(self.xp_rad, self.yp_rad, erfa.sp00(tt1, tt2))
        return erfa.c2tcio(erfa.c2ixys(x, y, s), erfa.era00(ut11, ut12), polar)

    def record(self):
        return {"utc": self.utc, "source": self.source,
                "available": self.available, "is_prediction": self.is_prediction,
                "days_predicted": self.days_predicted,
                "ut1_minus_utc_s": self.dut1_s, "ut1_sigma_s": self.ut1_sigma_s,
                "polar_motion_arcsec": (self.xp_rad / ARCSEC, self.yp_rad / ARCSEC),
                "polar_motion_sigma_arcsec": self.polar_motion_sigma_arcsec}


def catalog_columns_j2000(stars):
    """
    ERFA star-data columns (ra, dec, pmr, pmd, px, rv) at epoch J2000.0 for
    a list of CatalogStar. atco13 expects J2000.0 data and applies the
    space motion to the date itself.
    """
    dec = np.radians([s.dec_deg for s in stars])
    ra = np.radians([s.ra_deg for s in stars])
    pmr = np.array([s.pm_ra_cosdec_mas_yr for s in stars]) * MAS / np.cos(dec)
    pmd = np.array([s.pm_dec_mas_yr for s in stars]) * MAS
    px = np.array([s.parallax_mas for s in stars]) / 1000.0
    rv = np.array([s.radial_velocity_km_s for s in stars])
    e1, e2 = erfa.epj2jd(np.array([s.epoch_jyear for s in stars]))
    return erfa.pmsafe(ra, dec, pmr, pmd, px, rv, e1, e2, *J2000_JD)


def observing_context(eop, latitude_deg, longitude_deg, height_m, atmosphere):
    """
    ERFA star-independent astrometry parameters for one instant, site and
    atmosphere (erfa.apco13). atco13 and atoc13 build exactly this for every
    star; building it once and calling the "quick" transforms with it gives
    identical results much faster.
    """
    atmosphere = VACUUM if atmosphere is None else atmosphere
    astrom, _ = erfa.apco13(
        eop.utc_jd[0], eop.utc_jd[1], eop.dut1_s, np.radians(longitude_deg),
        np.radians(latitude_deg), height_m, eop.xp_rad, eop.yp_rad,
        atmosphere.pressure_hpa, atmosphere.temperature_c,
        atmosphere.relative_humidity, atmosphere.wavelength_um)
    return astrom


def observed_enu(columns, eop, latitude_deg, longitude_deg, height_m, atmosphere):
    """
    (N, 3) observed directions in local east/north/up for star columns at
    J2000.0: proper motion, parallax, light deflection, aberration, Earth
    rotation, diurnal effects and (pressure > 0) refraction. This is
    erfa.atco13, as ICRS -> CIRS (atciq) followed by CIRS -> observed (atioq).
    """
    astrom = observing_context(eop, latitude_deg, longitude_deg, height_m,
                               atmosphere)
    ra, dec, pmr, pmd, px, rv = columns
    ri, di = erfa.atciq(ra, dec, pmr, pmd, px, rv, astrom)
    az, zd, *_ = erfa.atioq(ri, di, astrom)
    sin_z = np.sin(zd)
    return np.column_stack((sin_z * np.sin(az), sin_z * np.cos(az), np.cos(zd)))


def icrs_from_observed_enu(enu, eop, latitude_deg, longitude_deg, height_m,
                           atmosphere):
    """
    (ra, dec) [rad] of the ICRS astrometric direction seen at given observed
    east/north/up directions: erfa.atoc13, the inverse of observed_enu for a
    star without space motion or parallax.
    """
    astrom = observing_context(eop, latitude_deg, longitude_deg, height_m,
                               atmosphere)
    enu = np.atleast_2d(enu)
    az = np.arctan2(enu[:, 0], enu[:, 1])
    zd = np.arctan2(np.hypot(enu[:, 0], enu[:, 1]), enu[:, 2])
    ri, di = erfa.atoiq("A", az, zd, astrom)
    return erfa.aticq(ri, di, astrom)


class ReferenceData:
    """
    catalog: {identity: CatalogStar}.
    eop_available=False models a device with no Earth-orientation data at
        all: UT1 is taken as UTC and the pole as the reference pole.
    eop_age_days declares how many days past its last measured value the
        Earth-orientation data in use is. The default reads it from the
        bundled table. The stated uncertainty grows with age (the growth law
        is the one recalled from IERS Bulletin A in Module 8; unverified).
    """

    def __init__(self, catalog, *, catalog_name="synthetic", eop_available=True,
                 eop_age_days=None):
        self.catalog = dict(catalog)
        self.catalog_name = catalog_name
        self.eop_available = bool(eop_available)
        self.eop_age_days = eop_age_days
        self._ids = list(self.catalog)
        self._index = {identity: i for i, identity in enumerate(self._ids)}
        stars = [self.catalog[i] for i in self._ids]
        for identity, star in zip(self._ids, stars):
            if not isinstance(star, CatalogStar):
                raise TypeError(f"catalog entry {identity!r} is not a CatalogStar")
        self._columns = (catalog_columns_j2000(stars) if stars
                         else tuple(np.zeros(0) for _ in range(6)))
        self._eop_cache = {}

    # -- catalog -------------------------------------------------------------
    def indices(self, star_ids):
        """(indices of known stars, boolean mask of which ids were known)."""
        known = np.array([i in self._index for i in star_ids], dtype=bool)
        index = np.array([self._index[i] for i, k in zip(star_ids, known) if k],
                         dtype=int)
        return index, known

    def _take(self, index):
        return tuple(column[index] for column in self._columns)

    def propagated(self, index, eop):
        """ICRS barycentric unit vectors at the date: space motion only."""
        ra, dec, *_ = erfa.pmsafe(*self._take(index), *J2000_JD, *eop.tdb_jd)
        return np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra),
                                np.sin(dec)))

    def observed(self, index, eop, latitude_deg, longitude_deg, height_m=0.0,
                 atmosphere=None):
        """
        (celestial (N, 3), enu (N, 3)): observed directions for an observer
        at a candidate position. `celestial` is the same direction in
        GCRS-oriented axes, the frame the camera attitude is solved in.
        """
        enu = observed_enu(self._take(index), eop, latitude_deg, longitude_deg,
                           height_m, atmosphere)
        to_celestial = eop.earth_from_celestial().T @ local_axes_ecef(
            latitude_deg, longitude_deg)
        return enu @ to_celestial.T, enu

    # -- Earth orientation -----------------------------------------------------
    def eop(self, utc):
        time = utc_time(utc)
        key = time.isot
        if key in self._eop_cache:
            return self._eop_cache[key]
        common = dict(utc=key, utc_jd=(float(time.jd1), float(time.jd2)),
                      tt_jd=(float(time.tt.jd1), float(time.tt.jd2)),
                      tdb_jd=(float(time.tdb.jd1), float(time.tdb.jd2)))
        status = iers_status(time)
        if self.eop_available and status["available"]:
            xp, yp = polar_motion(time)
            table = reference_status(time)
            days = table["days_predicted"] if self.eop_age_days is None \
                else float(self.eop_age_days)
            predicted = days > 0.0
            ut1_sigma, polar_sigma = prediction_uncertainty(days)
            record = EarthOrientationRecord(
                dut1_s=float(time.delta_ut1_utc), xp_rad=xp, yp_rad=yp,
                available=True, is_prediction=predicted, days_predicted=days,
                source=(status["source"] if self.eop_age_days is None else
                        f"{status['source']}, declared {days:.0f} days old"),
                ut1_sigma_s=max(ut1_sigma, MEASURED_UT1_SIGMA_S),
                polar_motion_sigma_arcsec=max(polar_sigma,
                                              MEASURED_POLAR_SIGMA_ARCSEC),
                **common)
        else:
            why = ("none loaded" if not self.eop_available
                   else f"none for this date ({status['source']})")
            record = EarthOrientationRecord(
                dut1_s=0.0, xp_rad=0.0, yp_rad=0.0, available=False,
                is_prediction=False, days_predicted=float("nan"),
                source=f"no Earth-orientation data ({why}): UT1 taken as UTC",
                ut1_sigma_s=NO_EOP_UT1_SIGMA_S,
                polar_motion_sigma_arcsec=NO_EOP_POLAR_SIGMA_ARCSEC, **common)
        self._eop_cache[key] = record
        return record

    def describe(self):
        import astropy
        import astropy_iers_data
        epochs = sorted({s.epoch_jyear for s in self.catalog.values()})
        return {"catalog": self.catalog_name, "catalog_stars": len(self.catalog),
                "catalog_epochs_jyear": epochs,
                "astronomy_library": f"ERFA (pyerfa {erfa.__version__})",
                "earth_orientation_table": "astropy-iers-data "
                                           f"{astropy_iers_data.__version__}",
                "astropy": astropy.__version__,
                "eop_available": self.eop_available,
                "eop_age_days_declared": self.eop_age_days,
                "downloads": "disabled"}
