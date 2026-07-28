import os

#folder = "some_folder"
#filename = "test_file.txt"

#path = os.path.join(folder, filename)
#print("Full path:", path)

file_object = open("filename.txt", "r")
#do something with file_object
file_object.close()

with open("filename.txt", "r") as file:
    content = file.read()
    print(content)

# read() looks at entire file contact as a single string
# readline() looks at one line at a time - better for large files
# readlines() looks at all lines in a file and returns them as a list of strings

with open("filename.txt", "r") as file:
    while True:
        line = file.readline()
        if not line: # no more lines to read
            break
        print("Line read:", line.strip())  # strip() removes leading/trailing whitespace

with open("filename.txt", "r") as file:
    lines = file.readlines()
    for index, line in enumerate(lines):
        print(f"Line {index + 1}:", line.strip())

print("File handling complete.\n \n")

with open("filename.txt", "r") as file:
    for line in file:
        print("Line read:", line.strip())

