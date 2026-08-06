class Pet():
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def speak(self):
        print(f"{self.name} makes a sound.")

class Cat(Pet):
    def speak(self):
        print(f"{self.name} says Meow!")

class Dog(Pet):
    def speak(self):
        print(f"{self.name} says Woof!")

class Rabbit(Pet):
    def speak(self):
        print(f"{self.name} says Squeak!")

pets = [Cat("Whiskers", 3), Dog("Fido", 5), Rabbit("Thumper", 2)]
for pet in pets:
    pet.speak()  # Calls the appropriate speak method based on the object's class