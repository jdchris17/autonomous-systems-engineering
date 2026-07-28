try:
    print("Great job,", name)

except:
    print("Error: name is not defined.")


try:
    name = "Zia"
    print("Great job,", name)
except:
    print("Error: name is not defined.")
else:
    print("No errors occurred.")

try:
    name = "Zia"
    print("Great job,", name)
except:
    print("Error: name is not defined.")
finally:
    print("This code will always run, regardless of whether an error occurred or not.")


try:
    with open("secret_file.txt", "r") as file:
        data = file.read()
        print(data)
except FileNotFoundError:
    print("Error: The file 'secret_file.txt' was not found.")
except PermissionError:
    print("Error: You do not have permission to read 'secret_file.txt'.")