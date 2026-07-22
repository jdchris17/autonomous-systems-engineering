i = 0
for j in range(11):
    i += j
print("The sum of the numbers from 0 to 10 is:", i)

name = ["Zia", "Wei", "Muchu"]
for i in name:
    print("Hello", i)

energy = 5
squirrel = 5
while energy > 0 and squirrel > 0:
    print("Wei chases the squirrel!")
    energy -= 1
    squirrel -= 1

if energy == 0 and squirrel == 0:
    print("Wei barely caught the squirrel.")
elif energy == 0 and squirrel > 0:
    print("Wei is too tired to catch the squirrel.")
elif energy > 0 and squirrel == 0:
    print("Wei caught the squirrel")