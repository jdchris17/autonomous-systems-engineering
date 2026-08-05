# creat a Plant class that can create plants with a name, size, desired climate, and alive status
# need to track how many have been created and update a counter reflecting that 
# define a count class attribute that keeps track of how many plants have been created
# define a water() method that prints a message and updates the alive status of the plant
# add a die() method that prints a dramatic message and updates the alive status of the plant
# make sure there's logic that can't let a plant die if it's already dead, and can't let a plant be watered if it's already dead

class Plant:
    count = 0  # class attribute to keep track of the number of plants created

    def __init__(self, name, size, desired_climate):
        self.name = name
        self.size = size
        self.desired_climate = desired_climate
        self.alive = True  # alive status of the plant
        Plant.count += 1  # increment the count when a new plant is created

    def water(self):
        if self.alive:
            print(f"You watered the {self.name}. It looks happy and healthy!")
        else:
            print(f"The {self.name} is dead and cannot be watered.")

    def die(self):
        if self.alive:
            self.alive = False
            print(f"Oh no! The {self.name} has died. It's a sad day for your garden.")
        else:
            print(f"The {self.name} is already dead.")

    def __str__(self):
        status = "alive" if self.alive else "dead"
        return f"{self.name} is a {self.size} plant that prefers {self.desired_climate} climate and is currently {status}." 
    
    # define a dunder method that tells python how to represent your plant as a string when you print it out, so that it prints out the name, size, desired climate, and alive status of the plant
    def __repr__(self):
        return f"Plant(name={self.name}, size={self.size}, desired_climate={self.desired_climate}, alive={self.alive})"
    


# Example usage:
plant1 = Plant("Rose", "Medium", "Temperate")
plant2 = Plant("Cactus", "Small", "Arid")

#example usage:
plant1.water()  # Output: You watered the Rose. It looks happy and healthy!
#example usage:
plant1.die()  # Output: Oh no! The Rose has died. It's a sad day for your garden.

print(f"Total plants created: {Plant.count}")  # Output: Total plants created: 2
#example usage:
print(plant1)  # Output: Rose is a Medium plant that prefers Temperate climate and is currently dead.