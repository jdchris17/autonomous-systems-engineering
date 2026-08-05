class ClothingItem():
    def __init__(self, garment_type, size, color, fabric_type):
        self.garment_type = garment_type.lower()
        self.size = size.lower()
        self.color = color.lower()
        self.fabric_type = fabric_type.lower()
        self.dirty = False
        self.times_worn = 0

    def wear(self):
        self.times_worn += 1
        print(f"Wearing {self.color} {self.garment_type}....")
        self.dirty = True

    def wash(self):
        if not self.dirty:
            print(f"{self.color} {self.garment_type} is clean.")
        else:
            print(f"This {self.garment_type} is all clean again!")

    def dye(self, new_color):
        self.color = new_color.lower()
        print(f"This {self.garment_type} is now {self.color}.")

    def __str__(self):
        return f"{self.color.capitalize()} {self.fabric_type} {self.garment_type}, size {self.size.capitalize()}"
    
    def __add__(self, other):
        combo = f"{self.color.capitalize()} {self.garment_type} + {other.color} {other.garment_type}"
        
        if self.garment_type == other.garment_type:
            verdict = "I don't think you need two of these"
        elif self.color == other.color:
            verdict = "matchy-matchy"
        elif self.garment_type == "jeans" and other.garment_type == "shirt" or self.garment_type == "shirt" and other.garment_type == "jeans":
            verdict = "classic combo"
        elif self.color in ["pink", "red"] and other.color in ["pink", "red"]:
            verdict = "disaster"
        else:
            verdict = "not bad"
        return f"{combo} = {verdict}"
    
    def __eq__(self, other):
        return (self.garment_type == other.garment_type and
                self.size == other.size and
                self.color == other.color)
    
shirt = ClothingItem("shirt", "M", "blue", "cotton")
jeans = ClothingItem("jeans", "M", "black", "denim")
jacket = ClothingItem("jacket", "L", "red", "leather")
copy_of_shirt = ClothingItem("shirt", "M", "blue", "cotton")

print(shirt)  # Output: Blue cotton shirt, size M
print(jacket)  # Output: Red leather jacket, size L

print("\n-- Wardrobe in use --")
shirt.wear()  # Output: Wearing blue shirt....
shirt.wear()  # Output: Wearing blue shirt....
print(f"Worn {shirt.times_worn} times.")

shirt.wash()  # Output: This shirt is all clean again!
shirt.wash()  # Output: This shirt is clean.

print("\n-- Getting crafty --")
jeans.dye("purple")  # Output: This jeans is now purple.
print(jeans)  # Output: Purple denim jeans, size M

print("\n-- Are these the same items? --")

print(shirt == copy_of_shirt)  # Output: True
print(shirt == jeans)  # Output: False

print("\n-- Fashion analysis --")
print(shirt + jeans)  # Output: Blue shirt + Black jeans = classic combo
print(shirt + jacket)  # Output: Blue shirt + Red jacket = not bad
print(shirt + copy_of_shirt)  # Output: Blue shirt + Blue shirt = I don't think you need two of these
    