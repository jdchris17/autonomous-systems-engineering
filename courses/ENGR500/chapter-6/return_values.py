def area_of_rectangle(length, width):
    area = length * width
    return area

result = area_of_rectangle(5, 3) #stores the value 15 in result
print("Area is:", result)

def return_nothing():
    print("Hi")

val = return_nothing() #stores the value None in val
print(val)

#interactive pet feeding with split functions
def get_pet_info():
    pet_name = input("What's your pet's name? ")
    
    if pet_name.lower() == 'quit':
        return None
    species = input("What species is your pet? (e.g., cat, dog):")
    food = input("What food would you like to feed them? ")
    
    return pet_name, species, food

def feed_pet(pet_name, species, food):
    if species.lower() == 'cat':
        print(f"{pet_name} arches their back and sniffs the {food}")
    elif species.lower() == 'dog':
        print(f"{pet_name} wags their tail at the site of {food}")
    else:
        print(f"{pet_name} looks curious about the {food}")

    print(f"Feeding {pet_name} some delicious {food} now!\n")

def interactive_feeding():
    print("Welcome to the interactive feeding program!")
    print("Type 'quit' when you're done feeding your pets.")

    while True:
        pet_data = get_pet_info()

        if pet_data is None: #user typed 'quit' for the pet's name
            print("All done feeding pets! Bye!")
            break

        pet_name, species, food = pet_data
        feed_pet(pet_name, species, food)

#call the main function "interactive_feeding()" to start the program

interactive_feeding()