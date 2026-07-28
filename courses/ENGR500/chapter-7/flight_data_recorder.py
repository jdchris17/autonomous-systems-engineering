import json

imu_packets = [
    {
        "time": 0.00,
        "roll": 0.10,
        "pitch": -0.02,
        "yaw": 182.1
    },
    {
        "time": 0.01,
        "roll": 0.11,
        "pitch": -0.01,
        "yaw": 182.0
    }
]

with open("imu_data.json", "w") as file:
    json.dump(imu_packets, file, indent=4)

with open("imu_data.json", "r") as file:
    loaded_imu_packets = json.load(file)
    for packet in loaded_imu_packets:
        print(f"Time: {packet['time']}\nRoll: {packet['roll']}\nPitch: {packet['pitch']}\nYaw: {packet['yaw']}\n")