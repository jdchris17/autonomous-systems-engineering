# by default everything is public
# use a leading underscore to show a method is private
# _secret_attribute as example

from unicodedata import name


class Cat:
    def __init__(self, name, age, color):
        self.name = name #public attribute
        self._age = age #private attribute
        self.color = color

# establishing getter and setter methods for the private attribute _age
class Cat:
    def __init__(self, name, age, color):
        self._name = name #private attribute
        self._age = age #private
        self.color = color

    def get_name(self):
        return self._name

    def set_name(self, new_name):
        if len(new_name) > 0:
            self._name = new_name
        else:
            print("Error: Invalid name.")