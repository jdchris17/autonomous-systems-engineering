#while condition:
    #code to execute repeatedly

count = 0
while count < 5:
    print("I love python!")
    count += 1

#count is just a stand in name. we could write anything like "x"

count = 0
while count < 5:
    print("Iteration", count + 1)
    print("count =", count)
    print("Condition", count, "< 5 is", count < 5)
    print("Print, 'I love python!'")
    count += 1
    print("count becomes", count)
    print("------------------------------")

print("count =", count)
print("Loop stops")

number = 5
while number > 0:
    print("Countdown:", number)
    number -= 1

print("Blast off!")