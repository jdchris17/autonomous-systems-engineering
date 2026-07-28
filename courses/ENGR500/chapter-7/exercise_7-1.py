import os

try:
    with open("letter.txt", "r") as file:
        content = file.read()
        print(content)
except FileNotFoundError:
    print("Error: The file 'letter.txt' was not found.")