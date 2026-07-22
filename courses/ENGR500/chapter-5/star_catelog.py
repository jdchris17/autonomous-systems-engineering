star_catalog = [
    {
        "name": "Polaris",
        "magnitude": 1.98,
    },
    {
        "name": "Sirius",
        "magnitude": -1.46,
    },
    {
        "name": "Canopus",
        "magnitude": -0.72,
    },
    {
        "name": "Arcturus",
        "magnitude": -0.05,
    },
    {
        "name": "Vega",
        "magnitude": 0.03,
    },
    {
        "name": "Capella",
        "magnitude": 0.08,
    },
]

for star in star_catalog:
    print(f"{star['name']}\nMagnitude: {star['magnitude']}\n---------\n")