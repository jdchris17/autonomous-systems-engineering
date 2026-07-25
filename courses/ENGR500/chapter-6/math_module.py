import math

x = math.sqrt(25)
print("The square root of 25 is:", x)

#math.pow(base, exponent) This function raises a number to a power

y = math.pow(2, 3)
print("2 raised to the power of 3 is:", y)

x = 5
math.factorial(x)
print("The factorial of", x, "is:", math.factorial(int(x)))

print(math.factorial(5))

angle_in_radians = math.radians(90) #converts degrees to radians
print(math.sin(angle_in_radians)) #sine of 90 degrees is 1

# math.log(x) is natural logarithm of x base e
# can use base-10 math.log10(x) or base-2 math.log2(x)

print(math.log(10)) #natural log of 10
print(math.log10(10)) #base-10 log of 10

print(math.ceil(4.2)) #rounds up to nearest integer
print(math.floor(4.8)) #rounds down to nearest integer  

print("Pi is approximately:", math.pi)
print("Euler's number, 'e', is approximately:", math.e)
