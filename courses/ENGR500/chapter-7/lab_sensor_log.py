from datetime import datetime

sensors = ["Camera", "IMU", "Clock"]

with open("sensor_log.txt", "w") as file:
    for i in range(3):  # Loop 10 times as an example
        for sensor in sensors:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"{timestamp} {sensor} OK\n"
            file.write(log_entry)
            print(log_entry.strip())