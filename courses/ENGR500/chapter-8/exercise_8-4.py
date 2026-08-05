# define a User class to create user objects.
# use decorators to log login and logout times for each user
# set up a constructor to set initial attributes: name, password, and loged_in status (always start as False)
#add getters and setters for each attribute - make sure password is at least 8 characters long and name isn't an empty string or spaces
#create a login() method that sets logged_in to True and logs the time of login
#create a logout() method that sets logged_in to False and logs the time of logout
#create a decorator to log actions to a file called user_log.txt, including the time of the action and the action itself (login or logout)

import time
from datetime import datetime

class User:
    def __init__(self, name, password):
        self._name = name
        self._password = password
        self._logged_in = False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if not value.strip():
            raise ValueError("Name cannot be empty or spaces.")
        self._name = value

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, value):
        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        self._password = value

    @property
    def logged_in(self):
        return self._logged_in

    def log_action(func):
        def wrapper(self, *args, **kwargs):
            action_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            action_name = func.__name__
            with open("user_log.txt", "a") as log_file:
                log_file.write(f"{action_time} - {self.name} performed {action_name}\n")
            return func(self, *args, **kwargs)
        return wrapper

    @log_action
    def login(self):
        self._logged_in = True
        print(f"{self.name} logged in.")

    @log_action
    def logout(self):
        self._logged_in = False
        print(f"{self.name} logged out.")

# Example usage:
user1 = User("Alice", "password123")
user1.login()  # Logs the login action
time.sleep(1)  # Simulate some time passing
user1.logout()  # Logs the logout action

# example users
user2 = User("Bob", "securepass")
user2.login()
user2.logout()

