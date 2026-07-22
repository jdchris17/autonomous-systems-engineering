workouts = {
    "Zia": {
        "bench press": {"weight": 5, "sets": 3, "reps": 10},
        "bicep curls": {"weight": 2, "sets": 3, "reps": 10},
        "box_jumps": {"weight": 0, "sets": 2, "reps": 10},
        "deadlifts": {"weight": 10, "sets": 2, "reps": 1},
        "kettlebell_swings": {"weight": 2, "sets": 3, "reps": 10},
        "lateral_raises": {"weight": 1, "sets": 3, "reps": 10},
        "stair_climber": {"minutes": 10}
    },
    "Wei": {
        "bench press": {"weight": 5, "sets": 1, "reps": 1},
        "leg_extensions": {"weight": 1, "sets": 2, "reps": 5},
        "planking": {"minutes": 5},
        "zoomies": {"minutes": 2}
    }
}

for member, exercises in workouts.items():
    print(f"{member}'s workout:")
    for exercise, details in exercises.items():
        if "minutes" in details:
            print(f"- {exercise}: {details['minutes']} minutes")
        else:
            print(f"- {exercise}: weight {details['weight']}lbs, {details['sets']} sets of {details['reps']} reps")
    print()  # Print a blank line between members

for member, exercises in workouts.items():
    total_weight = 0 #initialize total weight for each member
    for exercise, details in exercises.items():
        if "weight" in details: #only involve weight if exercise has weight
            total_weight += details["weight"] * details["sets"] * details["reps"]
    print(f"Total weight lifted by {member}: {total_weight} lbs")