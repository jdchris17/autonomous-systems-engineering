class Smoothie:
    # constructor to initialize the ingredients list
    def __init__(self, ingredients):
        self._ingredients = [ingredient.lower() for ingredient in ingredients]  # private attribute

    # @property decorator to deal with private attribute
    @property
    def ingredients(self):
        return self._ingredients
        
    #and a setter decorator to set the ingredients list
    @ingredients.setter
    def ingredients(self, new_ingredients):
        self._ingredients = new_ingredients

    #add a method to add ingredients to the list
    def add_ingredient(self, ingredient):
        ingredient = ingredient.lower()  # Convert to lowercase
        self.ingredients.append(ingredient)

    def describe(self):
        print("This smoothie contains the following ingredients:")
        for ingredient in self.ingredients:
            print(f"- {ingredient}")

    def __add__(self, other):
        combined = []
        for ingredient in self.ingredients + other.ingredients:
            if ingredient.lower() not in combined:
                combined.append(ingredient)
        return Smoothie(combined)
    

drink1 = Smoothie(["Banana"])
drink2 = Smoothie(["Strawberry", "apple"])

drink1.add_ingredient("Mango")
drink1.add_ingredient("apple")

drink3 = drink1 + drink2

drink3.describe()