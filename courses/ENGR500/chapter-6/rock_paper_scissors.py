# writing a program to play rock, paper, scissors with the user, where the user inputs their choice and the computer randomly selects its choice, and then the program determines the winner and prints the result.
# the program needs to show both the computer and user choices, and then print the result of the each round (win, lose, or tie).
# it will take 5 wins to win the game, and after each round, the program will print the current score to include ties, and after the winner is declared it will ask the user if they want to play again, and if they do, it will reset the score and start a new game.
# at any point, the user can type "quit" to exit the game.

def rock_paper_scissors():
    import random

    print("Welcome to Rock, Paper, Scissors!")
    print("Type 'rock', 'paper', or 'scissors' to play. Type 'quit' to exit.")

    user_score = 0
    computer_score = 0
    ties = 0

    while user_score < 5 and computer_score < 5:
        user_choice = input("Your choice: ").lower()
        if user_choice == "quit":
            print("Thanks for playing! Goodbye.")
            return

        if user_choice not in ["rock", "paper", "scissors"]:
            print("Invalid choice. Please try again.")
            continue

        computer_choice = random.choice(["rock", "paper", "scissors"])
        print(f"Computer chose: {computer_choice}")

        if user_choice == computer_choice:
            print("It's a tie!")
            ties += 1
        elif (user_choice == "rock" and computer_choice == "scissors") or \
             (user_choice == "paper" and computer_choice == "rock") or \
             (user_choice == "scissors" and computer_choice == "paper"):
            print("You win this round!")
            user_score += 1
        else:
            print("Computer wins this round!")
            computer_score += 1

        print(f"Score -> You: {user_score}, Computer: {computer_score}, Ties: {ties}")

    if user_score == 5:
        print("Congratulations! You won the game!")
    elif computer_score == 5:
        print("Computer won the game! Better luck next time.")

    play_again = input("Do you want to play again? (yes/no): ").strip().lower()
    if play_again == "yes":
        rock_paper_scissors()
    else:
        print("Thanks for playing! Goodbye.")

rock_paper_scissors()