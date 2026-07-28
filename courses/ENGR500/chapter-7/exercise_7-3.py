import os
import random

with open("note_to_self.txt", "w") as file:
    file.write("I must learn more python.\n")
    file.write("After this book I will do others.\n")
    file.write("The learning will never end.\n")
    file.write("I refuse to rely on AI.\n")

with open("note_to_self.txt", "r") as file:
    content = file.readlines()
    for index, line in enumerate(content):
        print(f"Line {index + 1}: {line.strip()}")