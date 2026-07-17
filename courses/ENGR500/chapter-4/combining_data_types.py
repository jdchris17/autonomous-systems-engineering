cat_attributes = {
    "name": "Zia",
    "age": 1,
    "color": "grey",
    "favorite_food": "tuna",
    "hobbies": ["napping", "programming", "playing"],
    "friends": [{"name": "Wei", "age": 7}]
}
print(cat_attributes["hobbies"])
print(cat_attributes["hobbies"][1])
print(cat_attributes["friends"][0]["name"])

#or different indentation
cat_attributes = {
    "name": "Zia",
    "age": 1,
    "color": "grey",
    "favorite_food": "tuna",
    "hobbies": ["napping", "programming", "playing"],
    "friends": [
        {
            "name": "Wei",
            "age": 7,
            "hobbies": ["napping", "visiting the forest"]
        },
        {
            "name": "Milo",
            "age": 3,
            "hobbies": [
                "running",
                "eating tuna",
                "scratching the couch"
            ]
        }
    ]
}
print(cat_attributes["friends"][0]["hobbies"][1])
print(cat_attributes["friends"][1]["hobbies"][2])
print(cat_attributes["friends"][1]["name"], "loves", cat_attributes["friends"][1]["hobbies"][0])
print(cat_attributes["friends"][1])
