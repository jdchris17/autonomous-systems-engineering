# variables inside a function are local by defualt

def feed_cat():
    food = "tuna" #local variable
    print("Feeding cat with", food)

feed_cat()
#print(food) #this will give an error because food is not defined outside the function

# variables outside a function are global by default

food = "kibble" #global variable

def feed_dog():
    print("Feeding dog with", food)

feed_dog()
print(food) #this will print "kibble" because food is defined outside the function

# global keyword allows you to modify a global variable inside a function

food = "kibble" #global variable

def feed_cat():
    global food #this tells Python that we want to use the global variable food
    food = "tuna" #this modifies the global variable food
    print("Feeding cat with", food)

feed_cat()
print(food) #this will print "tuna" because we modified the global variable food inside the function