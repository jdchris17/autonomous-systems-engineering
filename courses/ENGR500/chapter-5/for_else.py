toys = ["ball", "mouse", "string"]

for toy in toys:
    print("Checking:", toy)
    if toy == "feather":
        print("Found my feather!")
        break
else:
    print("No feather found. I'll nap.")


toys = ["ball", "feather", "string"]

for toy in toys:
    print("Checking:", toy)
    if toy == "feather":
        print("Found my feather!")
        break
else:
    print("No feather found. I'll nap.")


numbers = [1, 2, 3, 4, 5]
for num in numbers:
    if num % 2 == 0:
        numbers.remove(num)
    
print(numbers)

numbers = [1, 2, 2, 3, 4, 4, 5]
for num in numbers:
    if num == 2:
        numbers.remove(num)
print(numbers)

for num in numbers[:]: #[:] creates a copy of the list so that we can iterate over it while modifying the original list
    if num == 2:
        numbers.remove(num)
print(numbers)

