import math

print(math.dist((0, 0), (3, 4)))  # Should return 5.0

print(abs(7-13))  # Should return 6
print(math.dist((1, 2), (4, 6)))  # Should return 5.0

#return the avergae of a list of numbers

def average(numbers):
    total = sum(numbers)
    count = len(numbers)
    return total / count

print(average([1, 2, 3, 4, 5, 9, 23]))  # Should return 6.7

import statistics
print(statistics.mean([1, 2, 3, 4, 5, 9, 23, 32]))  # Should return 9.875


def is_sensor_enabled(sensor):
    return sensor.get("enabled", False)
sensor1 = {"name": "Temperature", "enabled": True}
sensor2 = {"name": "Humidity", "enabled": False}
sensor3 = {"name": "Pressure"}

print(is_sensor_enabled(sensor1))  # True
print(is_sensor_enabled(sensor2))  # False
print(is_sensor_enabled(sensor3))  # False