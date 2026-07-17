nr = 23454
if nr % 2 == 0:
    print("The number is even.")
else:
    print("The number is odd.")
nr = 5
answer = "The number is even." if nr % 2 == 0 else "The number is odd."
print(answer)

#next exercise
dog_name = "Wei"
dog_wtkg = 2.5
dog_wtlb = dog_wtkg * 2.20462
if dog_wtlb <= 12:
    dog_size = "Kaninchen"
elif dog_wtlb > 12 and dog_wtlb <= 16:
    dog_size = "Miniature"
elif dog_wtlb > 16 and dog_wtlb <= 32:
    dog_size = "Standard"
else:
    print("The dog is too big for this classification, and thus must not be a Dachshund.")
if dog_wtlb <= 32:
    print(f"With a weight of {dog_wtkg} kgs, {dog_wtlb} lbs, dachshund {dog_name} is classified as a {dog_size} Dachshund")

#next exercise
stored_password = "whiskers123"
input_password = input("Please enter your password: ")
if input_password == stored_password:
    print("Access granted.")
else:
    print("Access denied.")

#next exercise
nr_grade = 98
if nr_grade >= 95:
    letter_grade = "A+"
if nr_grade >= 90 and nr_grade < 95:
    letter_grade = "A"
if nr_grade >= 85 and nr_grade < 90:
    letter_grade = "B+"
if nr_grade >= 80 and nr_grade < 85:
    letter_grade = "B"
if nr_grade >= 70 and nr_grade < 80:
    letter_grade = "C"
if nr_grade >= 60 and nr_grade < 70:
    letter_grade = "D"
if nr_grade < 60:
    letter_grade = "F"
print(f"A numerical grade of {nr_grade} corresponds to a letter grade of {letter_grade}.")

#next exercise
day_of_wk = input("Please enter a day of the week: ")
match day_of_wk:
    case "Monday" | "monday":
        activity = "early morning run"
    case "Tuesday" | "tuesday":
        activity = "start a company"
    case "Wednesday" | "wednesday":
        activity = "early morning lift"
    case "Thursday" | "thursday":
        activity = "go to the beach after work"
    case "Friday" | "friday":
        activity = "go get dinner out"
    case "Saturday" | "saturday":
        activity = "early morning beach"
    case "Sunday" | "sunday":
        activity = "watch Redzone"
    case _:
        activity = "invalid day of the week"
print(f"On {day_of_wk}, you should {activity}.")