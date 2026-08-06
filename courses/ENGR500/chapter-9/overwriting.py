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
    def eat(self):
        print(f"{self.name} is eating from a marble bowl.")

# or

class Dog(Pet):
    def eat(self):
        super().eat()  # Call the parent class method
        print(f"{self.name} ...from a marble bowl.")  # Additional behavior

my_dog = Dog("Wei", 2.5, 1)
my_dog.eat()