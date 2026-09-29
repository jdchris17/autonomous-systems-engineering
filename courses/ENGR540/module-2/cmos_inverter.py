"""Module 2 -- a conceptual transistor-level CMOS inverter.

Not SPICE, and not an accurate MOSFET model (no saturation/triode
distinction, no body effect, no channel-length modulation). The point is
the *qualitative* transition: sweep an analog input voltage and watch a
digital inversion fall out of two switches with opposite thresholds.
That's the actual bridge this file exists to show -- everything in
Module 1 (NOT, NAND, ...) assumed 0/1 already existed. This is where
"0 and 1" comes from underneath: a continuous voltage, and two
transistors that disagree about which half of it counts as "on."

Circuit: a PMOS pulling Vout up toward VDD, an NMOS pulling Vout down
toward GND, gates tied together to the same Vin, drains tied together
at Vout.

    Vin low  (< Vtn)         -> NMOS off, PMOS on  -> Vout ~ VDD  (HIGH)
    Vin high (> VDD - Vtp)   -> NMOS on,  PMOS off -> Vout ~ GND  (LOW)
    Vin in between           -> both partially on  -> Vout in transition

That middle case is what actually produces the curve's slope, and it's
physically real: a real CMOS inverter briefly conducts current straight
from VDD to GND while switching, for exactly this reason (it's called
"shoot-through" current). It's not a modeling artifact to be embarrassed
about -- it's why the transfer curve is a smooth transition and not a
vertical cliff.

Model for "how on" each transistor is: linear in overdrive voltage
(gate-source voltage past the threshold), clipped at zero below
threshold. This is the standard small-signal approximation for a MOSFET
deep in its triode region (conductance is roughly proportional to
Vgs - Vt there) -- simplified, but not arbitrary. Output voltage is then
a resistive-divider between the two "conductances," same intuition as a
voltage divider made of two resistors whose values happen to depend on
Vin.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VDD = 5.0   # V, supply rail (classic textbook 5V CMOS logic level)
VTN = 1.0   # V, NMOS threshold
VTP = 1.0   # V, PMOS threshold magnitude


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def nmos_conductance(vin, vtn=VTN):
    """NMOS pull-down 'on-ness'. Gate-source voltage is just Vin (source
    tied to GND). Off below threshold; conducts more, the further above
    threshold the gate is driven.
    """
    return np.maximum(0.0, vin - vtn)


def pmos_conductance(vin, vdd=VDD, vtp=VTP):
    """PMOS pull-up 'on-ness'. Source is tied to VDD, so the relevant
    overdrive is measured from the *top* rail down: (VDD - Vin) is how
    hard the gate is being pulled below the source. Off above
    VDD - Vtp; conducts more as Vin drops toward GND.
    """
    return np.maximum(0.0, (vdd - vin) - vtp)


def inverter_output(vin, vdd=VDD, vtn=VTN, vtp=VTP):
    """Approximate Vout as a resistive divider between the pull-up
    (PMOS) and pull-down (NMOS) conductances: whichever transistor is
    "more on" pulls Vout closer to its own rail.

    If both are off at once (only possible when vtn + vtp > vdd, not
    the case for this module's default thresholds), the output floats;
    that's modeled here as VDD/2 rather than a division by zero, flagged
    explicitly rather than silently -- it should not happen for the
    thresholds this module ships with.
    """
    g_n = nmos_conductance(vin, vtn)
    g_p = pmos_conductance(vin, vdd, vtp)
    total = g_n + g_p
    total_safe = np.where(total > 0, total, 1.0)  # avoid /0; result discarded by the outer where
    return np.where(total > 0, vdd * g_p / total_safe, vdd / 2)


def transistor_states(vin, vdd=VDD, vtn=VTN, vtp=VTP):
    """Human-readable ON/OFF state of each transistor at a given Vin."""
    nmos_on = vin > vtn
    pmos_on = vin < (vdd - vtp)
    return ("ON" if nmos_on else "OFF"), ("ON" if pmos_on else "OFF")


def digital_level(v, vdd=VDD):
    """Interpret a voltage as a digital LOW/HIGH using the VDD/2 midpoint
    -- the same interpretation a downstream logic gate would apply.
    """
    return "HIGH" if v > vdd / 2 else "LOW"


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def sweep(n=500, vdd=VDD, vtn=VTN, vtp=VTP):
    """Vin swept LOW to HIGH across the full rail; returns (vin, vout)."""
    vin = np.linspace(0.0, vdd, n)
    vout = inverter_output(vin, vdd, vtn, vtp)
    return vin, vout


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def main():
    vin, vout = sweep()

    print("=" * 72)
    print("CONCEPTUAL CMOS INVERTER -- transistor-level mini-simulation")
    print("=" * 72)
    print(f"VDD = {VDD:.1f} V   Vtn = {VTN:.1f} V   Vtp = {VTP:.1f} V")
    print()
    print(f"{'Vin (V)':>8}  {'NMOS':>4}  {'PMOS':>4}  {'Vout (V)':>9}  "
          f"{'Vin digital':>11}  {'Vout digital':>12}")
    sample_points = [0.0, VTN / 2, VTN, VDD / 2, VDD - VTP, VDD - VTP / 2, VDD]
    for v in sample_points:
        n_state, p_state = transistor_states(v)
        vo = float(inverter_output(v))
        print(f"{v:8.2f}  {n_state:>4}  {p_state:>4}  {vo:9.3f}  "
              f"{digital_level(v):>11}  {digital_level(vo):>12}")
    print()
    print("Every row away from the exact switching point inverts: Vin digital")
    print("HIGH gives Vout digital LOW and vice versa -- the analog sweep")
    print("produces a digital NOT gate as an emergent property of two")
    print("opposite-threshold switches, not something hand-coded in. The one")
    print("exception is Vin = VDD/2 itself: by symmetry Vout also lands on")
    print("VDD/2, which is precisely the voltage this table can't classify")
    print("as LOW or HIGH -- a real inverter is similarly indeterminate at")
    print("the instant it's mid-switch, which is why real digital logic is")
    print("never operated in that middle region, only swept through it.")
    print("=" * 72)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(vin, vout, color="tab:red", lw=2, label="Vout(Vin)")
    ax.axvline(VDD / 2, color="gray", ls=":", lw=1, label="VDD/2 (switching threshold)")
    ax.axhline(VDD / 2, color="gray", ls=":", lw=1)
    ax.plot([0, VDD / 2], [VDD, VDD], color="tab:blue", ls="--", lw=1, alpha=0.6,
            label="ideal digital inverter")
    ax.plot([VDD / 2, VDD / 2], [VDD, 0], color="tab:blue", ls="--", lw=1, alpha=0.6)
    ax.plot([VDD / 2, VDD], [0, 0], color="tab:blue", ls="--", lw=1, alpha=0.6)
    ax.set_xlim(0, VDD)
    ax.set_ylim(-0.2, VDD + 0.2)
    ax.set_xlabel("Vin (V)")
    ax.set_ylabel("Vout (V)")
    ax.set_title("CMOS inverter transfer curve (conceptual model)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left")
    fig.tight_layout()
    fig.savefig("cmos_inverter.png", dpi=120)


if __name__ == "__main__":
    main()
    plt.show()
