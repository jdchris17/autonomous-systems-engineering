remaining_treats = 200

def print_treats_left():
    print(f"You have {remaining_treats} treats left.")

def eat_treats(remaining, eaten_treats):
    print(f"You ate {eaten_treats} treats.")
    remaining -= eaten_treats
    return remaining

print_treats_left()
remaining_treats = eat_treats(remaining_treats, 50)
print_treats_left()