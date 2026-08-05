class Cat:
    def __init__(self, name, age, color):
        self.name = name
        self.age = age
        self.color = color

    def __str__(self):
        return f"{self.name} is {self.age} years old, color={self.color}"
    
cat = Cat("Zia", 3, "black")
print(cat)  # Output: Cat(name=Zia, age=3, color=black)

# __len__(self) method returns the length of the object. For example, if you have a class that represents a collection of items, you can define the __len__() method to return the number of items in that collection.

class ToyBox:
    def __init__(self, toys):
        self.toys = toys

    def __len__(self):
        return len(self.toys)
    
box = ToyBox(["ball", "feather", "laser pointer"])
print(len(box))  # Output: 3

# __eq__(self, other) method is used to compare two objects for equality. You can define this method to specify how two instances of your class should be compared.

class Cat:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        return self.name == other.name and self.age == other.age
        
cat1 = Cat("Zia", 3)
cat2 = Cat("Zia", 3)
cat3 = Cat("Muchu", 5)

print(cat1 == cat2)  # Output: True
print(cat1 == cat3)  # Output: False


class Cat:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __eq__(self, other):
        return True
    
cat1 = Cat("Zia", 3)
cat2 = Cat("Zia", 3)
cat3 = Cat("Muchu", 5)

print(cat1 == cat2)  # Output: True
print(cat1 == cat3)  # Output: True because we defined __eq__ to always return True, regardless of the attributes of the objects being compared.

# __add__(self, other) method is used to define the behavior 
# of the addition operator (+) for instances of your class. 
# You can implement this method to specify how two objects 
# should be combined when the + operator is used.

class Cat:
    def __init__(self, name):
        self.name = name

    def __add__(self, other):
        return [self, other]
    
cat1 = Cat("Zia")
cat2 = Cat("Muchu")
family = cat1 + cat2  # This will call the __add__ method
print([cat.name for cat in family])  # Output: ['Zia', 'Muchu']