fav_animals = [
"dog",
"lion",
"red panda"
]
print("I have", len(fav_animals), "favorite animals.")
most_fav = fav_animals[0]
print("The", most_fav, "is my most favorite animal.")
print("The", fav_animals[2], "is my least favorite of my favorite animals.")

fav_nap_spots = ["window", "cardboard box", "humans lap", "luxery cat bed"]
del fav_nap_spots[3]
fav_nap_spots.insert(0, "keyboard")
fav_nap_spots.insert(2, "sunbeam on the floor")
fav_nap_spots.append("Wei's bed")
if "human bed" not in fav_nap_spots:
    fav_nap_spots.append("human bed")
print("Current length of list is:", len(fav_nap_spots))
print("My favorite nap spots are:", fav_nap_spots)
top_3 = fav_nap_spots[0:3]
print("My top 3 favorite nap spots are:", top_3)
fav_nap_spots[1], fav_nap_spots[3] = fav_nap_spots[3], fav_nap_spots[1]
print("My top 3 favorite nap spots are now:", fav_nap_spots[0:3])