# building a ClothingItem class that has attributes for type of clothing, size, color, and fabric type
# we'll define dye() method that changes the color of the clothing item
# we'll define a __str__, so printing a ClothingItem returns something readable
# we'll use another dunder method to see if two clotthing items are the same type, color, and size
# we'll add a wear() method that prints a message and keeps track of how many times a garments been word, and make it mark them as dirty
# add a wash() method that checks whether something needs to be washed and prints a fitting message
# create a feature that prints messages when you add up two garments to tell you if one garment plus another is a good fashion move or bad

class ClothingItem:
    def __init__(self, clothing_type, size, color, fabric_type):
        self.clothing_type = clothing_type
        self.size = size
        self.color = color
        self.fabric_type = fabric_type
        self.times_worn = 0
        self.is_dirty = False

    def dye(self, new_color):
        self.color = new_color

    def wear(self):
        self.times_worn += 1
        self.is_dirty = True
        print(f"You wore the {self.color} {self.clothing_type}. It's now dirty.")

    def wash(self):
        if self.is_dirty:
            print(f"You washed the {self.color} {self.clothing_type}. It's now clean.")
            self.is_dirty = False
            self.times_worn = 0
        else:
            print(f"The {self.color} {self.clothing_type} is already clean.")

    def __str__(self):
        return f"{self.size} {self.color} {self.fabric_type} {self.clothing_type}"

    def __eq__(self, other):
        return (self.clothing_type == other.clothing_type and
                self.size == other.size and
                self.color == other.color)

    def __add__(self, other):
        if (self.clothing_type == "shirt" and other.clothing_type == "pants") or (self.clothing_type == "pants" and other.clothing_type == "shirt"):
            return f"The {self.color} {self.clothing_type} and the {other.color} {other.clothing_type} make a good outfit!"
        else:
            return f"The {self.color} {self.clothing_type} and the {other.color} {other.clothing_type} do not match well."  
        
# Example usage:
shirt = ClothingItem("shirt", "M", "blue", "cotton")
pants = ClothingItem("pants", "M", "black", "denim")

print(shirt + pants)  # Output: The blue shirt and the black pants make a good outfit!

#example usage:
shirt.wear()  # Output: You wore the blue shirt. It's now dirty.
shirt.wash()  # Output: You washed the blue shirt. It's now clean.

#example usage:
shirt.dye("red")
print(shirt)  # Output: M red cotton shirt