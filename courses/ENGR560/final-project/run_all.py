"""Run the capstone end to end (~6 minutes on CPU): Experiment A (integration + full pytest suite),
Experiment B (verification / explicit failure), Experiment C (classical vs hybrid systems).
The threshold calibration (experiments/calibrate_thresholds.py, ~3 min) is not re-run here; its result is
recorded in AcceptanceConfig and outputs/calibrate_thresholds_log.txt."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    for name in ("exp_a_integration", "exp_b_verification", "exp_c_comparison"):
        print("=" * 110, flush=True)
        log = ROOT / "outputs" / f"{name.split('_')[0]}_{name.split('_')[1]}_log.txt"
        p = subprocess.run([sys.executable, str(ROOT / "experiments" / f"{name}.py")], capture_output=True, text=True)
        print(p.stdout, flush=True)
        log.write_text(p.stdout)
        if p.returncode:
            print(p.stderr)
            sys.exit(p.returncode)
