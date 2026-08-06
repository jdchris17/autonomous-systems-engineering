class Camera:
    def __init__(self, name, resolution, field_of_view):
        self.name = name
        self.resolution = resolution
        self.field_of_view = field_of_view

    def capture_image(self):
        print(f"{self.name} is capturing an image with {self.resolution} resolution and {self.field_of_view} field of view.")

    def calibrate(self):
        print(f"{self.name} is calibrating the camera.")

    def display_status(self):
        print(f"Camera Name: {self.name}")
        print(f"Resolution: {self.resolution}")
        print(f"Field of View: {self.field_of_view}")


class IMU:
    def __init__(self, roll, pitch, yaw):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def update(self):
        print(f"IMU is updating with Roll: {self.roll}, Pitch: {self.pitch}, Yaw: {self.yaw}")
    
    def reset(self):
        print("IMU is resetting to default orientation.")

    def display_orientation(self):
        print(f"Current Orientation - Roll: {self.roll}, Pitch: {self.pitch}, Yaw: {self.yaw}")


class Battery:
    def __init__(self, charge, voltage):
        self.capacity = charge
        self.voltage = voltage

    def check_status(self):
        print(f"Battery Status - Charge: {self.capacity}%, Voltage: {self.voltage}V")

    def charge(self):
        print("Battery is charging.")

    def discharge(self):
        print("Battery is discharging.")

# now create NavigationSytems class that has a Camera, IMU, and Battery as attributes. 
# then display_system_status method calls camera.display_status(), imu.display_orientation(), and battery.check_status() methods.

class NavigationSystem:
    def __init__(self, camera, imu, battery):
        self.camera = camera
        self.imu = imu
        self.battery = battery

    def display_system_status(self):
        self.camera.display_status()
        self.imu.display_orientation()
        self.battery.check_status()

# Example usage
camera = Camera("Front Camera", "1080p", "120 degrees")
imu = IMU(1.3, 0.5, 180.0)
battery = Battery(85, 12.6)

navigation_system = NavigationSystem(camera, imu, battery)
navigation_system.display_system_status()