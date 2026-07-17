my_first_dictionary = {}
cat_attributes = {
    "name": "Zia",
    "age": 1,
    "color": "grey",
    "favorite_food": "tuna"
}
cat_name = cat_attributes["name"]
print("Cat's name is:", cat_name)
print("Cat's age is:", cat_attributes["age"])
breed = cat_attributes.get("breed")
print("Breed:", breed)
breed = cat_attributes.get("breed", "unknown")
print("Breed:", breed)

#adding or updating a key-value pair in the dictionary
cat_attributes["hobby"] = "napping"
cat_attributes["age"] = 2
print("Updated cat attributes:", cat_attributes)

#deleting a key-value pair from the dictionary
del cat_attributes["favorite_food"]
print("After deleting favorite_food:", cat_attributes)

hobby = cat_attributes.pop("hobby")
print("Removed hobby:", hobby)
print("After removing hobby:", cat_attributes)

#breed = cat_attributes.pop("breed") will return error
breed = cat_attributes.pop("breed", "Not Specified")
print("Removed breed:", breed)