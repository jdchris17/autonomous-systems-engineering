class Plant():
    count = 0  # class attribute to keep track of the number of plants created

    def __init__(self, name, size, desired_climate):
        self.name = name
        self.size = size
        self.desired_climate = desired_climate
        self.alive = True  # alive status of the plant
        Plant.count += 1  # increment the count when a new plant is created

    def water(self):
        if not self.alive:
            print(f"The {self.name} is dead and cannot be watered.")
        else:
            print(f"You watered the {self.name}. It looks happy and healthy!")

    def die(self):
        if not self.alive:
            print(f"The {self.name} is already dead.")
        else:
            print(f"Oh no! The {self.name} has died. It's a sad day for your garden.")
            self.alive = False
            Plant.count -= 1  # decrement the count when a plant dies

    def __str__(self):
        return f"{self.name} is a {self.size} plant that prefers {self.desired_climate} climate and is currently {'alive' if self.alive else 'dead'}."
    
plant1 = Plant("Rose", "Medium", "Temperate")
plant2 = Plant("Cactus", "Small", "Arid")
plant3 = Plant("Tulip", "Small", "Temperate")

print(f"Plant count: {Plant.count}")  # Output: Plant count: 3

plant1.water()  # Output: You watered the Rose. It looks happy and healthy!
plant1.die()  # Output: Oh no! The Rose has died. It's a sad day for your garden.

print(f"Plant count: {Plant.count}")  # Output: Plant count: 2

plant1.water()  # Output: The Rose is dead and cannot be watered.

print(plant2)
print(plant1)  # Output: Rose is a Medium plant that prefers Temperate climate and is currently dead.
