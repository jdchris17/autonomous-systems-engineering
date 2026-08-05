# @property decorator allows us to define a method that can be accessed like an attribute

class Cat:
    def __init__(self, name):
        self._name = name  # private attribute

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, new_name):
        if len(new_name) > 0:
            self._name = new_name
        else:
            print("Error: Invalid name.")

my_cat = Cat("Zia")
print(my_cat.name)  # this calls the getter method
my_cat.name = "Max"  # this calls the setter method
print(my_cat.name)  # this calls the getter method
my_cat.name = ""  # this calls the setter method and will print an error message
