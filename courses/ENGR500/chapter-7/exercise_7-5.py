import json

def load_woods():
    try:
        with open("wood_types.json", "r") as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print("The JSON file does not exist.")

def display_woods():
    woods = load_woods()
    for wood in woods:
        uses = ""
        for i in range(len(wood["uses"])):
            uses += wood["uses"][i]
            if i < len(wood["uses"]) - 1:
                uses += ", "
        print(f"Name: {wood["name"]}")
        print(f"Hardness: {wood["hardness"]}")
        print(f"Color: {wood["color"]}")
        print(f"Uses: {uses}")
        print()  # Add an empty line between entries

display_woods()

# now add some wood

def add_wood(name, hardness, color, uses):
    new_wood = {
        "name": name,
        "hardness": hardness,
        "color": color,
        "uses": uses
    }

    woods = load_woods()
    woods.append(new_wood)

    with open("wood_types.json", "w") as file:
        json.dump(woods, file)

add_wood("Dark Oak", 8, "dark brown", ["flooring", "furniture", "cabinets", "architectural trim"])

def filter_wood(use):
    woods = load_woods()
    print(f"Woods suitable for {use}:")
    for wood in woods:
        if use in wood["uses"]:
            print(wood["name"])

filter_wood("furniture")