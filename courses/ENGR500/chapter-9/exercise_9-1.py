class Vehicle:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

# we're going to make child classes for Car, Truck, and Motorcycle that inherit from Vehicle
# add a ride() method that can be applied to every type of vehicle. Print a message "[vehicle type] is riding"
    def ride(self):
        print(f"{self.__class__.__name__} is riding.")

    def __str__ (self):
        return f"{self.year} {self.make} {self.model}"
    
class Car(Vehicle):
    def __init__(self, make, model, year, num_doors):
        super().__init__(make, model, year)
        self.num_doors = num_doors

    def __str__(self):
        return f"{super().__str__()} with {self.num_doors} doors"

class Truck(Vehicle):
    def __init__(self, make, model, year, bed_length):
        super().__init__(make, model, year)
        self.bed_length = bed_length

    def __str__(self):
        return f"{super().__str__()} with a bed length of {self.bed_length} feet"

class Motorcycle(Vehicle):
    def __init__(self, make, model, year, has_sidecar):
        super().__init__(make, model, year)
        self.has_sidecar = has_sidecar

    def __str__(self):
        sidecar_status = "with a sidecar" if self.has_sidecar else "without a sidecar"
        return f"{super().__str__()} {sidecar_status}"
    
vehicles = [
    Car("Toyota", "Camry", 2020, 4),
    Truck("Ford", "F-150", 2019, 6.5),
    Motorcycle("Harley-Davidson", "Sportster", 2021, False)
]
for vehicle in vehicles:
    print(vehicle)
    vehicle.ride()
