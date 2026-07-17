#slicing allows to get a subset of the list
cat_toys = ["feather", "mouse toy", "ball", "laser pointer", "scratching post"]
favorite_toys = cat_toys[1:4]
print("Favorite toys:", favorite_toys)
#the slicing goes UP TO index 4 but not included
#only 1, 2, 3 are included

most_toys = cat_toys[1:5]
#print("Most toys:", most_toys)
most_toys = cat_toys[1:]
#print("Most toys:", most_toys)

number_of_toys = len(cat_toys)
print("I have", number_of_toys, "toys.")
print("Computed differently, I have", len(cat_toys), "toys.")

item_exists = "laser pointers" in cat_toys

if "laser pointer" in cat_toys:
    print("Time to play with the laser pointer!")
else:
    print("No laser pointer to play with.")

if item_exists:
    print("REALLY time to play with the laser pointer!")
else:
    print("REALLY no laser pointer to play with.")

cat_toys.append("cardboard box")
if "cardboard box" in cat_toys:
    print("Oh no, she's throwing away my cardboard box!")
    cat_toys.remove("cardboard box")
else:
    print("Nothing to throw away, I guess.")