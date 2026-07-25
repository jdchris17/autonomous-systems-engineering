imu_packet = {
    "timestamp": 12.4,
    "roll": 0.1,
    "pitch": 0.2,
    "yaw": 0.3
}

def validate_packet(packet, required_keys):
    for key in required_keys:
        if key not in packet:
            return False
    return True

# now print the result of True or False for each key in the imu_packet
required_keys = ["timestamp", "roll", "pitch", "yaw"]
for key in required_keys:
    if key in imu_packet:
        print(f"{key} True")
    else:
        print(f"{key} False")