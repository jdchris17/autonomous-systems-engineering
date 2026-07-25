import random

def fetch(name, item):
    responses = [
        f"{name} fetched the {item}!",
        f"{name} ran and grabbed the {item}!",
        f"{name} brought back the {item}!",
        f"{name} is a good fetcher! Here's the {item}!",
        f"{name} is ignoring the {item}.",
        f"{name} is too tired to fetch the {item}."
    ]
    response = random.choice(responses)
    print(response)

fetch("Wei", "ball")