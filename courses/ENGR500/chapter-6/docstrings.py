def area_of_circle(radius):
    """Calculate the area of a circle given its radius.
    Uses formula: area = pi * radius^2
    """
    import math
    return math.pi * radius ** 2

def get_pet_info():
    """Prompt the user for their pet's name, species, and food.
    Returns a tuple (pet_name, species, food) or None if the user types 'quit'.
    """
    pet_name = input("What's your pet's name? ")
    
    if pet_name.lower() == 'quit':
        return None
    species = input("What species is your pet? (e.g., cat, dog):")
    food = input("What food would you like to feed them? ")
    
    return (pet_name, species, food)

def feed_pet(pet_name, species, food):
    """Feed the pet based on its species and the food provided.
    Prints a message indicating how the pet reacts to the food.
    """
    if species.lower() == 'cat':
        print(f"{pet_name} arches their back and sniffs the {food}")
    elif species.lower() == 'dog':
        print(f"{pet_name} wags their tail at the site of {food}")
    else:
        print(f"{pet_name} looks curious about the {food}")

    print(f"Feeding {pet_name} some delicious {food} now!\n")

def interactive_feeding():
    """Main function to run the interactive pet feeding program.
    Continuously prompts the user for pet information until they type 'quit'.
    """
    print("Welcome to the interactive feeding program!")
    print("Type 'quit' when you're done feeding your pets.")

    while True:
        pet_data = get_pet_info()

        if pet_data is None:  # user typed 'quit' for the pet's name
            print("All done feeding pets! Bye!")
            break

        pet_name, species, food = pet_data
        feed_pet(pet_name, species, food)

#call the main function "interactive_feeding()" to start the program
interactive_feeding()