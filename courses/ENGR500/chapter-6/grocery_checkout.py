# writing a program that simulates a grocery checkout process, where the user can input the number of items they're buying and the prices of each, and the program will calculate the total cost and print a receipt

def grocery_checkout():
    print("Welcome to the grocery checkout program!")
    num_items = int(input("Enter the number of items you're buying: "))
    total_cost = 0
    for i in range(num_items):
        price = float(input(f"Enter the price of item {i+1}: "))
        total_cost += price
    print("\nReceipt")
    print("-------")
    print(f"Total cost: ${total_cost:.2f}")

grocery_checkout()