imu_packet = {
    "time": 125.43,
    "roll": 0.25,
    "pitch": -1.7,
    "yaw": 182.4
}
print(imu_packet)
imu_packet["roll"] = 0.5
print(imu_packet)
imu_packet["altitude"] = 23
print(imu_packet)
del imu_packet["pitch"]
print(imu_packet)
if "pitch" in imu_packet:
    print("Pitch is in the packet", imu_packet["pitch"])
else:
    imu_packet["pitch"] = -1.5
    print("Pitch is not in the packet, so I added it back in with a value of", imu_packet["pitch"])
print(imu_packet)