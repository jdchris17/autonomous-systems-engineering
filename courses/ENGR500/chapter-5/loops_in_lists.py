cat_toys = ["feather", "ball", "laser pointer", "scratching post"]

print("Listing all my cat toys:")
for toy in cat_toys:
    print("-", toy)

coordinates = (36.34, 76.45)
print("Coordinates are:")
for x in coordinates: #could say "for coordinate in coordinates" or "for c in coordinates"
    print(x)

cat_attributes = {
    "name": "Zia",
    "age": 3,
    "color": "grey",
    "fav_food": "tuna"
}

print("Cat attributes are:")
for key in cat_attributes: #could say "for k in cat_attributes"
    print("-", key)

print("Cat attributes fully are:")
for key in cat_attributes:
    print("-", key, ":", cat_attributes[key])

print("Cat attributes are:")
for value in cat_attributes.values(): #could say "for x in cat_attributes.values()"
    print("-", value)

print("Cat attributes and values:")
for key, value in cat_attributes.items(): #could say "for k, v in cat_attributes.items()"
    print(f"- {key} : {value}")