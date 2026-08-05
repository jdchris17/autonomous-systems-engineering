def some_decorator(func):
    def wrapper():
        print("Something happens Before the function call")
        func()
        print("Something happens After the function call")
    return wrapper

def my_function():
    print("This is my function.")

my_function = some_decorator(my_function)
my_function()

#like a function that takes a function as an argument and returns a new function that adds some behavior before and after the original function is called.
#now we can use the @ syntax to apply the decorator to a function more cleanly:

def some_decorator(func):
    def wrapper():
        print("Something happens Before the function call")
        func()
        print("Something happens After the function call")
    return wrapper

@some_decorator
def my_function():
    print("This is my function.")
my_function()