cat_attributes = {
    "name": "Zia",
    "age": 1,
    "color": "grey",
    "favorite_food": "tuna"
}
#keys() returns a list of the keys in the dictionary without the values
keys = cat_attributes.keys()
print("Keys:", list(keys))

#values() returns a list of the values in the dictionary without the keys
values = cat_attributes.values()
print("Values:", list(values))
print("Values differently:", values)

#items() returns all the key-value pairs in the dictionary as a list of tuples
items = cat_attributes.items()
print("Items:", list(items))

if "age" in cat_attributes:
    print("I know my age!")
    print("It's ", cat_attributes["age"])