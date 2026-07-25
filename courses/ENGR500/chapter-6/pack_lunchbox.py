# writing a function to pack a lunchbox with a main course, snack, and a drink
# need user input for each item, and then print out the packed lunchbox contents
# also need to return a reminder "Forgot to pack a snack." if a snack or any other input is ignored

def pack_lunchbox(main_course, snack, drink):
    if not main_course:
        main_course = "Forgot to pack a main course."
    if not snack:
        snack = "Forgot to pack a snack."
    if not drink:
        drink = "Forgot to pack a drink."
    
    print("Packing lunchbox with:")
    print(f"Main Course: {main_course}")
    print(f"Snack: {snack}")
    print(f"Drink: {drink}")
    
    return "Lunchbox packed successfully!"

def main():
    print("Welcome to the lunchbox packing program!")
    main_course = input("Enter the main course for the lunchbox: ")
    snack = input("Enter the snack for the lunchbox: ")
    drink = input("Enter the drink for the lunchbox: ")
    
    result = pack_lunchbox(main_course, snack, drink)
    print(result)

main()