contacts = {
    "Avery Darrah": 8754,
    "Tom Evans": 3487,
    "Jarrett Darrah": 4460
}
contacts["Tom Gorman"] = 9972
if "Maaike" not in contacts:
    contacts["Maaike"] = 1234
contacts.pop("Imke", None)
print(list(contacts.keys()))

library = [
    {
        "Title:": "Jane Eyre",
        "Author:": "Charlotte Bronte, 1816, British",
        "Page count:": 659,
        "Publication year:": 1847
    },
    {
        "Title:": "Python Illustrated",
        "Author:": "Zia van Putten",
        "Page count:": 432,
        "Publication year:": 2026
    }
]

library_new = {
    "Jane Eyre": {
        "author": [{"name": "Charlotte Bronte", "birth_year": 1816, "nationality": "British"}],
        "page_count": 659,
        "publication_year": 1847
    },
    "Python Illustrated": {
        "author": [{"name": "Zia van Putten", "birth_year": 2026, "nationality": "American"}],
        "page_count": 432,
        "publication_year": 2026
    }
}
print(list(library_new.keys()))
if "John Adam" in library_new["Python Illustrated"]["author"]:
    print("John Adam is an author of Python Illustrated")
else:
    print("John Adam is not an author of Python Illustrated")

#print nationality of the author of Jane Eyre
print(library_new["Jane Eyre"]["author"][0]["nationality"])