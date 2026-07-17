#tuples are immutable lists, meaning they cannot be changed after they are created. They are defined using parentheses instead of square brackets.
my_first_tuple = ()
position = (10, 20)
cat_info = ("Zia", 1, "grey")

#single item tuple still have to have a comma test = ("thing",)
name = cat_info[0]
age = cat_info[1]
color = cat_info[2]
print("Name:", name)
print("Age:", age)
print("Color:", color)

#unpacking (works with tuples) [and with lists]
dog_info = ("First", 5, "white")
name, age, color = dog_info
print("Name:", name)
print("Age:", age)
print("Color:", color)