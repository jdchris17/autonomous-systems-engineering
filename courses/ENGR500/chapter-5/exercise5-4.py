friends = ["Max", "Louise", "Jacob", "Sam", "Farah", "Selim"]
team_a = []
team_b = []

for index in range(len(friends)):
    if index % 2 == 0:
        team_a.append(friends[index])
    else:
        team_b.append(friends[index])

print("Team A:")
for friend in team_a:
    print(friend)
print("\nTeam B:")
for friend in team_b:
    print(friend)

nr = 52
factorial = 1
for i in range(1, nr + 1):
    factorial *= i
print("The factorial of", nr, "is:", factorial)