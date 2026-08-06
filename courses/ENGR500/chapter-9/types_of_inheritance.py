#single
#multilevel
#hierarchical
#multples

#single inheritance
class Human:
    pass

class PetOwner(Human):
    pass

#multilevel inheritance - parent class -> child class -> grandchild class

class Pet:
    def eat(self):
        print("Pet is eating.")

class RobotPet(Pet):
    def recharge(self):
        print("Robot pet is recharging.")

class RoboDog(RobotPet):
    def bark(self):
        print("RoboDog is barking.")

#hierarchical inheritance - one parent class -> multiple child classes
class Cat(Pet):
    pass

class Dog(Pet):
    pass

class Rabbit(Pet):
    pass

#multiple inheritance - child class inherits from multiple parent classes
# possible to call methods that have the same name in different parent classes 
# Method Resolution Order (MRO) determines the order in which methods are inherited from parent classes.
# The MRO is determined by whichever parent class is listed first in the child class definition. If a method is called on an object of the child class, Python will first look for that method in the child class itself. If it doesn't find it there, it will then look in the parent classes in the order they are listed in the child class definition.
class LoginUser:
    def login(self, username, password):
        print(f"{username} logged in.")

class Speaker:
    def __init__(self, name, sound):
        self.name = name
        self.sound = sound

    def speak(self):
        print(f"{self.name} is speaking about {self.sound}")

class LoggedInSpeaker(Speaker, LoginUser):
    def upload_slides(self, slides):
        pass

s = LoggedInSpeaker("Dr. Zia", "Quantum Computing")
s.login("Dr. Zia", "password123")  # Output: Dr. Zia logged in.
s.speak()  # Output: Dr. Zia is speaking about Quantum Computing