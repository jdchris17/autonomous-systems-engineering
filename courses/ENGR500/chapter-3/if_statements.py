hour = 18
if hour == 18:
    print("It's 6 PM, time for dinner.")
a = 5
b = 5
if a == b:
    print("a and b are equal.")
a = 5
b = 10
if a != b:
    print("a and b are not equal.")
jake_wt = 225
avery_wt = 125
if jake_wt > avery_wt:
    print("Jake is heavier than Avery.")
    print("Avery is lighter than Jake.")
nap_hr = 2
if nap_hr >= 3:
    print("That is too long")
if nap_hr < 3:
    print("Good, otherwise it would be too long")
is_hungry = False
if is_hungry:
    print("Eat something")
else:
    print("Don't eat anything")
weather = "rainy"
if weather == "sunny":
    print("Wear sunglasses")
elif weather == "rainy":
    print("Bring an umbrella")
elif weather == "cloudy":
    print("No sunscreen needed")
else:
    print("Check the weather forecast")
is_sunny = True
is_wife_home = True
going_outside = is_sunny and is_wife_home
if going_outside:
    print("Going outside with wife")
else:
    print("Gotta wait until she's home")
if is_sunny and is_wife_home:
    print("Going outside with wife again")
is_raining = False
if not is_raining:
    print("It's not raining, so I can go outside")