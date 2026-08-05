# create a Smoothie class where you can add ingredients, print all ingredients, and even mix two together using the + operator. Use dunder methods to implement this functionality.
# mark ingredients private
# when program is run ask the user to input ingredients, include a way to stop adding ingredients and print the list of ingredients. Then ask the user if they want to input another smoothie to mix with the first one and print the list of ingredients for the mixed smoothie.

class Smoothie:
    def __init__(self):
        self._ingredients = []  # private attribute

    def add_ingredient(self, ingredient):
        if ingredient:  # Check if the ingredient is not empty
            self._ingredients.append(ingredient)
        else:
            print("Error: Invalid ingredient.")

    def get_ingredients(self):
        return self._ingredients

    def __add__(self, other):
        mixed_smoothie = Smoothie()
        mixed_smoothie._ingredients = self._ingredients + other._ingredients
        return mixed_smoothie
    
# Example usage:
if __name__ == "__main__":
    smoothie1 = Smoothie()
    while True:
        ingredient = input("Enter an ingredient for the first smoothie (or type 'done' to finish): ")
        if ingredient.lower() == 'done':
            break
        smoothie1.add_ingredient(ingredient)

    print("Ingredients in the first smoothie:", smoothie1.get_ingredients())

    smoothie2 = Smoothie()
    while True:
        ingredient = input("Enter an ingredient for the second smoothie (or type 'done' to finish): ")
        if ingredient.lower() == 'done':
            break
        smoothie2.add_ingredient(ingredient)

    print("Ingredients in the second smoothie:", smoothie2.get_ingredients())

    mixed_smoothie = smoothie1 + smoothie2
    print("Ingredients in the mixed smoothie:", mixed_smoothie.get_ingredients())