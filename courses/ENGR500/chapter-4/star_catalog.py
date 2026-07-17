stars = [
    "Polaris",
    "Sirius",
    "Betelgeuse",
    "Vega",
    "Rigel"
]
stars.append("Capella")
stars.insert(2, "Procyon")
stars.remove("Rigel")
print(stars)
stars.sort()
print(stars)
stars.reverse()
print(stars)
some_stars = stars[1:4]
print(some_stars)