import os

def load_sensor_log(filename):
    
# from this function, If the file exists: Return and print its contents. If it doesn't: Print an informative message and return an empty list or None. This is our first exposure to defensive programming.

    if os.path.exists(filename):
        with open(filename, "r") as file:
            content = file.read()
            print(content)
            return content
    else:
        print(f"Error: The file '{filename}' was not found.")
        return None
    
load_sensor_log("sensor_log.txt")

load_sensor_log("non_existent_file.txt")