# Inheritance helps avoid code duplication by allowing classes to inherit attributes and methods from other classes. This promotes code reusability and a cleaner, more organized codebase.
#class ChildClass(ParentClass):
    # child class body
# class Dog(Pet):
    #pass

class Pet:
    def __init__(self, name, weight, age):
        self.name = name
        self.weight = weight
        self.age = age

    def eat(self):
        print(f"{self.name} is eating.")

    def run(self):
        print(f"{self.name} is running.")

    def play(self):
        print(f"{self.name} is playing.")

    def breathe(self):
        print(f"{self.name} is breathing.")

class Dog(Pet):
    def fetch(self):
        print(f"{self.name} is fetching the ball.")

# add new attributes to the child class that are not in the parent class
#use the super() function to call the parent class constructor and initialize the inherited attributes

class Cat(Pet):
    def __init__(self, name, age, weight, number_of_lives_left):
        super().__init__(name, weight, age)  # Call the parent class constructor
        self.number_of_lives_left = number_of_lives_left

    def climb(self):
        print(f"{self.name} is climbing the tree.")

class Rabbit(Pet):
    def hop(self):
        print(f"{self.name} is hopping around.")

my_dog = Dog("Wei", 2.5, 1)
my_dog.fetch()  # Output: Wei is breathing.