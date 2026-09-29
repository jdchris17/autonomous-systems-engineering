# Module 2 — Conceptual transistor-level CMOS inverter

Optional, per the spec: a simplified conceptual CMOS inverter, not SPICE.
The goal isn't an accurate MOSFET model — it's the qualitative bridge
between an analog input voltage and the digital inversion that [Module 1](../module-1/README.md)'s
`logic.py` took for granted (`NOT`, `NAND`, ... all assumed 0/1 already
existed as inputs). This is where 0 and 1 come from underneath.

## Files

| File | What it does |
|---|---|
| [`cmos_inverter.py`](cmos_inverter.py) | PMOS pulls `Vout` toward `VDD`, NMOS pulls it toward GND, both gated by the same `Vin`. Each transistor's "on-ness" is modeled as a simple linear overdrive above its threshold (`Vin - Vtn` for the NMOS, `(VDD - Vin) - Vtp` for the PMOS), clipped at zero below threshold — the standard small-signal approximation for a MOSFET's triode region. `Vout` is then a resistive divider between the two: `Vout = VDD * g_pmos / (g_pmos + g_nmos)`. Sweeps `Vin` from 0 to `VDD`, prints a sample table of transistor states and digital LOW/HIGH interpretation, and plots the transfer curve against an ideal instantaneous-step inverter for comparison. |

## Why the curve has a slope at all

With `Vtn = Vtp = 1.0 V` and `VDD = 5.0 V`, there's a region (`1 V < Vin <
4 V`) where **both** transistors conduct at once. That's not a modeling
artifact — real CMOS inverters do this too (it's called shoot-through
current), and it's exactly what produces the transfer curve's slope
instead of a vertical cliff. Outside that region, one transistor is
firmly off and the other pulls `Vout` cleanly to its rail.

## The one edge case worth knowing about

At exactly `Vin = VDD/2`, the circuit is symmetric and `Vout` also lands
on `VDD/2` — the one voltage the LOW/HIGH digitizer in this file can't
classify either way. That's not a bug to fix; it's the real inverter's
own instability point mid-switch, which is why digital logic is never
*operated* in that middle region, only swept through it on the way
between states. `cmos_inverter.py`'s sample table calls this out
explicitly rather than papering over it.

## Run it

```
python cmos_inverter.py
```

Prints a state table (`Vin`, NMOS/PMOS on-off, `Vout`, digital
interpretation of both) and writes `cmos_inverter.png`, the transfer
curve.

## Status

Optional and self-contained — doesn't feed into `cpu_simulator` yet. Next
up, per the course plan: gates → adders → ALU, building on Module 1's
`logic.py`.
