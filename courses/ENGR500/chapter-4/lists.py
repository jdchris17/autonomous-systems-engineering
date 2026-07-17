#an empty list
my_first_list = []
cat_toys = ["feather", "ball", "laser pointer", "scratching post"]
first_toy = cat_toys[0]
print("First toy", first_toy)
print("Second toy", cat_toys[1])
print("Third toy", cat_toys[2])
print("Fourth toy", cat_toys[3])
print("First toy but a differnt way", cat_toys[-4])
print("Last toy but a differnt way", cat_toys[-1])

#append adds an item to the end of the list
cat_toys.append("cardboard box")
print("Updated cat toys list", cat_toys)

#insert adds an item to a specific index in the list
cat_toys.insert(1, "mouse toy")
print("After inserting at index 1:", cat_toys)
print(len(cat_toys))

#remove removes first occurrence of an item in the list (and only the first)
cat_toys.remove("ball")
print("After removing the 'ball':", cat_toys)

#pop removes an item at a specific index and returns it
cat_toys.pop(2)
print("After popping the item at index 2:", cat_toys)

#clear removes all items from the list
cat_toys.clear()
print("After clearing the list:", cat_toys)

#we can use the index to overwrite an item in the list
cat_toys = ["feather", "ball", "laser pointer", "scratching post"]
print("Before overwriting the item at index 1:", cat_toys)
cat_toys[1] = "catnip toy"
print("After overwriting the item at index 1:", cat_toys)