def greet():
    print("Hello! Welcome to the program.")

greet()

def greet_user(name):
    print(f"Hello, {name}! Welcome to the program.")

greet_user("Zia")
greet_user("Wei")

def feed_pet(pet_name, food):
    print(f"Feeding {pet_name} with {food}.")

feed_pet("Zia", "tuna")
feed_pet("Wei", "salmon")

def interactive_feeding():
    print("Welcome to the interactive feeding program!")
    print("Type 'quit' when you're done feeding your pets.")

    while True:
        pet_name = input("What's your pet's name? ")

        if pet_name.lower() == 'quit':
            print("All done feeding pets! Bye!")
            break

        species = input("What species is your pet? (e.g., cat, dog):")
        food = input("What food would you like to feed them? ")

        if species.lower() == 'cat':
            print(f"{pet_name} arches their back and sniffs the {food}")
        elif species.lower() == 'dog':
            print(f"{pet_name} wags their tail and happily eats the {food}")
        else:
            print(f"{pet_name} looks curious about the {food}")

        print(f"Feeding {pet_name} some delicious {food} now!\n")

interactive_feeding()