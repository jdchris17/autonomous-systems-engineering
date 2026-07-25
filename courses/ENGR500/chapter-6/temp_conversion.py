def temp_convert():
    temp_c = float(input("Enter temperature in Celsius: "))
    temp_f = (temp_c * 9/5) + 32
    print(f"{temp_c} degrees Celsius is equal to {temp_f} degrees Fahrenheit.")

def temp_convertf():
    temp_f = float(input("Enter temperature in Fahrenheit: "))
    temp_c = (temp_f - 32) * 5/9
    print(f"{temp_f} degrees Fahrenheit is equal to {temp_c} degrees Celsius.")

def temp_convermain():
    choice = input("Convert from Celsius or Fahrenheit? (C/F): ").strip().upper()
    if choice == 'C':
        temp_convert()
    elif choice == 'F':
        temp_convertf()
    else:
        print("Invalid choice. Please enter 'C' or 'F'.")

temp_convermain()

#next we will convert feet to meters and meters to feet.

def feet_to_meters():
    feet = float(input("Enter length in feet: "))
    meters = feet * 0.3048
    print(f"{feet} feet is equal to {meters} meters.")

def meters_to_feet():
    meters = float(input("Enter length in meters: "))
    feet = meters / 0.3048
    print(f"{meters} meters is equal to {feet} feet.")

def length_convertmain():
    choice = input("Convert from Feet or Meters? (F/M): ").strip().upper()
    if choice == 'F':
        feet_to_meters()
    elif choice == 'M':
        meters_to_feet()
    else:
        print("Invalid choice. Please enter 'F' or 'M'.")

length_convertmain()