with open("log.txt", "w") as file:
    file.write("This is a log entry.\n")
    file.write("Another log entry.\n")

with open("log.txt", "a") as file:
    file.write("Appending a new log entry.\n")

# CSV files - comma-separated values
import csv

with open("pets.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Type", "Age"])
    writer.writerow(["Buddy", "Dog", 5])
    writer.writerow(["Mittens", "Cat", 3])

with open("pets.csv", "a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Charlie", "Dog", 2])

with open("pets.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

#now to get just the columns we want, we use an index or we can use the csv.DictReader class

with open("pets.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row[0])  # prints the first column (Name)
