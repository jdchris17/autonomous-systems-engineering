from datetime import datetime

now = datetime.now()
print("Right now, the date and time is:", now)

current_time = datetime.now()
print("Year:", current_time.year)
print("Month:", current_time.month)
print("Day:", current_time.day)
print("Hour:", current_time.hour)
print("Minute:", current_time.minute)
print("Second:", current_time.second)


from datetime import datetime, timedelta

future_date = datetime.now() + timedelta(days=7)
print("One week from now, the date and time will be:", future_date)