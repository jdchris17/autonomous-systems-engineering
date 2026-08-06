# create a Superhero class and each instance of Superhero will have multiple powers or tolls that an instance of Superhero can use
# the Superhero class will have attributes for name, power, and gadget
# each power and gadget will be simple classes with one method each, a use() method
# first creat a few power and gadget classes, then create a Superhero class. its constructor will initialize attributes fo rname, powers, and gadgets
# when creating the instance of Superhero, you pass in a list of powers and gadgets
# add a __Sstr__ method that introduces the superhero and lists their powers and gadgets
# add a fight_crime() method that will randomly select a power and gadget to call use() on

import random

class SuperStrength:
    def use(self):
        print("Using super strength!")

class Flight:
    def use(self):
        print("Flying through the sky!")

class Invisibility:
    def use(self):
        print("Becoming invisible!")

class Telekinesis:
    def use(self):
        print("Using telekinesis!")

class GrapplingHook:
    def use(self):
        print("Using grappling hook!")

class LaserGun:
    def use(self):
        print("Firing laser gun!")

#two more gadgets
class SmokeBomb:
    def use(self):
        print("Using smoke bomb!")

class JetPack:
    def use(self):
        print("Using jet pack!")

class Superhero:
    def __init__(self, name, powers, gadgets):
        self.name = name
        self.powers = powers
        self.gadgets = gadgets

    def __str__(self):
        power_names = [power.__class__.__name__ for power in self.powers]
        gadget_names = [gadget.__class__.__name__ for gadget in self.gadgets]
        return f"Superhero {self.name} has powers: {', '.join(power_names)} and gadgets: {', '.join(gadget_names)}"

    def fight_crime(self):
        power = random.choice(self.powers)
        gadget = random.choice(self.gadgets)
        print(f"{self.name} is fighting crime!")
        power.use()
        gadget.use()

# create a superhero with a list of powers and gadgets
my_hero = Superhero("Captain Code", [SuperStrength(), Flight(), Invisibility()], [GrapplingHook(), LaserGun(), SmokeBomb()])
print(my_hero)

my_hero.fight_crime()  # Call the fight_crime method to see the superhero in action