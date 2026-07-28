import csv
import os

def count_participants():
    with open("wiener_dog_race_results.csv", "r") as file:
        reader = csv.reader(file)
        data = list(reader)[1:]  # Skip the header row
        
        #determine amount of participants using len() function

        print(f"There are {len(data)} participants in the race: ")

        #dealing with empty rows in the csv file

        counter = 0
        for row in data:
            if len(row) >=2 and row[0].strip() and row[1].strip():  # Check if the row is empty
                counter += 1

        print(f"{counter} participants have valid data in the CSV file.") 

count_participants()

def add_participant(name, time):
    name_exists = False
    results = [name, time]

    try:
        with open("wiener_dog_race_results.csv", "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) and row[0] == name: # Check if the name already exists
                    name_exists = True
                    break
        
        if not name_exists:
            with open("wiener_dog_race_results.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(results)
                print(f"Participant {name} added successfully.")
        else:
            print(f"Participant {name} already exists.")
    except FileNotFoundError:
        print("The CSV file does not exist.")

add_participant("Wei", "42.0")

#rank dogs based on their race times

def rank_dogs():
    try:
        with open("wiener_dog_race_results.csv", "r") as file:
            reader = csv.reader(file)
            data = list(reader)[1:]  # Skip the header row
            
            # Filter out rows with empty name or time
            data = [row for row in data if len(row) >= 2 and row[0].strip() and row[1].strip()] # only valid entires

            for i in range(len(data)):
                data[i][1] = float(data[i][1])  # Convert time to float for sorting

            for i in range(len(data)):
                for j in range(i + 1, len(data)):
                    if data[i][1] > data[j][1]:  # Compare times
                        data[i], data[j] = data[j], data[i]  # Swap rows

        with open("wiener_dog_race_results_sorted.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(['Name', 'Finish Time (seconds)'])  # Write header
            writer.writerows(data)  # Write sorted data

    except FileNotFoundError:
        print("The CSV file does not exist.")

rank_dogs()

#determine winner

def determine_winner():
    try:
        with open("wiener_dog_race_results_sorted.csv", "r") as file:
            reader = csv.reader(file)
            data = list(reader)
            winner = data[1]  # The first row after the header is the winner
            print(f"The winner is {winner[0]} with a time of {winner[1]} seconds.")
    except FileNotFoundError:
        print("The CSV file does not exist.")


determine_winner()