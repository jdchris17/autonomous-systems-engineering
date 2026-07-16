like_cats = "no"
cat_person = like_cats == "yes"
print("Is the person a cat person? ", cat_person)
joke_prompt = "Why did Wei sit in the shade?"
joke_answer = "Because she didn't want to be a \"hot dog!\""
print(joke_prompt,"\n",joke_answer)
food_per_day = 65
total_food = 1500
days_of_food = total_food // food_per_day
leftover_food = total_food % food_per_day
print(f"The number of days of food Wei has is: {days_of_food}. The amount left over is: {leftover_food} grams.")
day_of_month = 15
fifth = day_of_month % 5
adventure_day = fifth == 0
print("Is it an adventure day?", adventure_day)
fun_fact = "I love reading"
print("My fun fact is this many characters:", len(fun_fact))
loud_fact = fun_fact.upper()
quiet_fact = fun_fact.lower()
print("My fun fact in all caps is:", loud_fact)
print("My fun fact in all lowercase is:", quiet_fact)