visible_stars = [
    "Polaris",
    "Vega",
    "Altair",
    "Deneb",
    "Arcturus"
]
visible_stars.append("Sirius")
visible_stars.insert(2, "Betelgeuse")
avail_stars = visible_stars.pop(3)
print(visible_stars)
print(f"The available star that was removed is: {avail_stars}.")
visible_stars.insert(2, "Altair")
del visible_stars[4]
print(sorted(visible_stars))
print(len(visible_stars))