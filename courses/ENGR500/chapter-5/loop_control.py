for number in range(1, 10):
    if number == 5:
        print("Found the number 5! Exiting loop.")
        break
    print("Current number:", number)
print("Loop has ended.")

while True:
    text = input("Do you want to stop? ")
    if(text == "yes"):
        break

for number in range(1, 6):
    if number == 3:
        print("Skipping number...")
        continue
    print("Current number:", number)