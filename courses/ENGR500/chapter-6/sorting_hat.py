# writing a function that asks for your name and then sorts you into a Hogwarts house randomly

import random
import time

def sorting_hat():
    name = input("What is your name? ")
    houses = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    house = random.choice(houses)
    print("Sorting you into a house...")
    time.sleep(3)  # pause for 2 seconds to simulate sorting
    print(f"{name}, you have been sorted into... {house}!")

sorting_hat()