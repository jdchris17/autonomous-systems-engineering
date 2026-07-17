mission_phases = [
    "Capture Sky",
    "Plate Solve",
    "Determine Altitude",
    "Estimate Postition",
    "Display Navigation Solution"
]
# index tracks the number, item tracks the mission phase
for index, item in enumerate(mission_phases, start=1):
    print(f"Mission Phase {index}. {item}")