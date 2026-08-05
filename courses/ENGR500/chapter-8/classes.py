class MyClass:
    pass  # Placeholder for class implementation

#class Cat:
    #def greet(self): # the first parameter of a method is always self.
     #   print("Meow!")

#my_cat = Cat()
#my_cat.greet()  # Output: Meow!

class Cat:
    def __init__(self, name, age, color): #the constructor
        self.name = name
        self.age = age
        self.color = color

    def greet(self):
        print(f"{self.name} says Meow!")

my_cat = Cat("Zia", 3, "black")
my_cat.greet()  # Output: Zia says Meow!
print(my_cat.age)
my_cat.age = 4
print(my_cat.age) 
print(my_cat.color)
print(my_cat.name)


#when init is used you have to provide the arugments when creating an instance
#to get around that you can set default values like "None"

class Cat:
    def __init__(self, name=None, age=None, color=None): #the constructor
        self.name = name
        self.age = age
        self.color = color

    def greet(self):
        print(f"{self.name} says Meow!")

my_cat1 = Cat()
my_cat1.greet()  # Output: Meow!
my_cat2 = Cat("Zia", 3, "black")
my_cat2.greet()  # Output: Zia says Meow!


#you can make a fixed attribute to a class by defining it outside 'init'
class Cat:
    species = "Felis catus"  # Class attribute

    def __init__(self, name, age, color): #the constructor
        self.name = name
        self.age = age
        self.color = color
