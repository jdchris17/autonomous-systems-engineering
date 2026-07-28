import os

# Check if file exists; only write header if it's new
if not os.path.exists("attendance.txt"):
    with open("attendance.txt", "w") as file:
        file.write("This is the guest list for the event.\n")

name = input("Please enter your name: ")
with open("attendance.txt", "a") as file:
    file.write(f"{name}\n")
    print(f"Thank you, {name}. Your name has been added to the attendance list.")
