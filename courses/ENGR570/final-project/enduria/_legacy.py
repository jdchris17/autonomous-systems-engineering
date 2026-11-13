"""
The one place where the Module 1-8 folders are put on the import path.

The module folders stay where they are as the educational reference and as
the single source of every algorithm. This package does not copy them: the
subject modules of `enduria` (timekeeping, frames, attitude, gravity,
geodesy, estimation, astrometry) re-export the validated functions under
one namespace, and everything new in the capstone imports from those.

Importing this module also applies Module 2's offline policy: no
Earth-orientation or leap-second data is ever downloaded.
"""

import sys
from pathlib import Path

COURSE_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIRS = [COURSE_ROOT / f"module-{n}" for n in range(1, 9)]

_missing = [str(d) for d in MODULE_DIRS if not d.is_dir()]
if _missing:
    raise ImportError("enduria needs the course module folders next to "
                      f"final-project; missing: {_missing}")

for _directory in reversed(MODULE_DIRS):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import astronomical_time as _astronomical_time  # noqa: E402,F401  (offline IERS)
