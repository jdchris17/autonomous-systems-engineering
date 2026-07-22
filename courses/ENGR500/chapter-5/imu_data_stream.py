imu_samples = [
    0.12,
    0.09,
    0.08,
    -0.02,
    0.05,
    0.18,
    -0.10
]

for sample in imu_samples:
    if abs(sample) > 0.10:
        print(f"WARNING. \nAngular velocity, {sample}, exceeds threshold.")


t = 10
while t > 0:
    print(f"T-{t}")
    t -= 1
print("Launch")

for i in range(10, 0, -1):
    print(f"T-{i}")
print("Launch")