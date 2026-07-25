import random

roll = random.randint(1, 6)
print("Dice roll:", roll)

#random.random() generates a random float between 0.0 and 1.0
value = random.random()
print("Random float between 0.0 and 1.0:", value)

pets = ["cat", "dog", "hamster", "parrot"]
chosen_pet = random.choice(pets)
print("Randomly chosen pet:", chosen_pet)

toys = ["ball", "feather", "string"]
random.shuffle(toys)
print("Shuffled toys:", toys)

#random.sample(sequence, k) returns a list of k unique elements chosen from the sequence
sampled_toys = random.sample(toys, 2)
print("Randomly sampled toys:", sampled_toys)