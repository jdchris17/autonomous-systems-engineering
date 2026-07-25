# the 'type' function tells you the data type of a variable
# the int(), float(), and str() functions convert data types

items = ["feather", "ball", "laser pointer"]
count = len(items) #len() function returns the number of items in a list
type_count = type(count) #type() function returns the data type of a variable
print("Count:", count)
print("Type of count:", type_count)
count_text = str(count) #str() function converts a variable to a string
type_count_text = type(count_text)
print("Count:", count_text)
print("Type of count:", type_count_text)