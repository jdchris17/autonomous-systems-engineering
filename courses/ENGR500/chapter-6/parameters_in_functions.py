# default parameters

def greet_friend(name, greeting="Hello"):
    print(greeting, name)
    
greet_friend("Zia")
greet_friend("Wei", "Hi")

# keyword arguments / parameters
def describe_pet(pet_name, species, age):
    print(f"{pet_name} is a {age}-year-old {species}.")

describe_pet(species="dachshund", pet_name="Wei", age=7)

# variable-length arguments
def show_off(*args, **kwargs):
    print("Positional arguments:", args)
    print("Keyword arguments:", kwargs)

show_off("Zia", "Wei", toy="feather wand", snack="tuna")