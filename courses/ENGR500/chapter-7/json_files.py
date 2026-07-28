#JSON - JavaScript Object Notation
# represents data as dictionaries and lists in text format
#values can be objects, arrays, strings, numbers, booleans, or null

#outer brackets {} represent an object (dictionary)
#inner brackets [] represent an array (list)

{
    "name": "Zia",
    "species": "cat",
    "age": 3,
    "favorite_foods": ["tuna", "chicken", "catnip"],
    "best_friend": {
        "name": "Wei",
        "species": "dog",
        "age": 5,
        "favorite_foods": ["beef", "peanut butter"]
    }
}

{
    "pets": [
        {
            "name": "Zia",
            "species": "cat",
            "age": 3,
            "favorite_foods": ["tuna", "chicken", "catnip"]
        },
        {
            "name": "Wei",
            "species": "dog",
            "age": 5,
            "favorite_foods": ["beef", "peanut butter"]
        },
        {
            "name": "Axel",
            "species": "axolotl",
            "age": 2,
            "favorite_foods": ["earthworms", "pellets"]
        }
    ]
}

import json

data = {
    "pet": "Zia",
    "toys": ["ball", "feather", "laser pointer"],
    "weight": 4.5,
}

with open("pet_data.json", "w") as file:
    json.dump(data, file)

with open("pet_data.json", "r") as file:
    loaded_data = json.load(file)
    print(loaded_data["toys"])

