pet_type = "axolot1"
match pet_type:
    case "cat":
        print("A cat goes meow.")
    case "dog":
        print("A dog goes woof.")
    case "parrot":
        print("A parrot says 'Polly wants a cracker!'")
    case "hamster":
        print("A hamster squeaks.")
    case "snake":
        print("A snake goes hiss!")
    case _:
        print("I don't know what sound that pet makes.")

command = "sit"
match command:
    case "sit":
        print("The dog sits down.")
    case "stay":
        print("The dog stays in place.")
    case "roll over":
        print("The dog rolls over.")

pet_type = "puppy"
match pet_type:
    case "cat" | "kitten":
        print("A cat goes meow.")
    case "dog" | "puppy":
        print("A dog goes woof.")