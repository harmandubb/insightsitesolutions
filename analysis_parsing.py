import re
from datetime import datetime, time
import numpy as np

def extract_video_parameters(file_name):
    # Define the regex pattern to match the desired data
    pattern = r"Average_TIMER_PER_FRAME:\s([\d.]+)\sDATE:\s(\d{2}-\d{2}-\d{4})\sSTART_TIME:\s(\d{2}:\d{2}:\d{2})"
    
    # Open the file and read its content
    with open(file_name, 'r') as file:
        content = file.read()

    # Use regex to search for the pattern
    match = re.search(pattern, content)
    
    if match:
        # Extract the values from the match object
        average_time_per_frame = float(match.group(1))
        date_str = match.group(2)  # Extracted date as string
        time_str = match.group(3)  # Extracted time as string

        # Convert date_str and time_str to datetime and time objects
        date = datetime.strptime(date_str, "%m-%d-%Y").date()  # Convert to date object
        start_time = datetime.strptime(time_str, "%H:%M:%S").time()  # Convert to time object
        
        return average_time_per_frame, date, start_time
    else:
        raise ValueError("Could not find the necessary data in the file.")
    
    import numpy as np
import ast

def analysis_data_extract(line):
    """
    Parses a file where each line contains a number and a list, and stores it into a NumPy array.
    
    Args:
    file_name (str): The name or path of the file to read.
    
    Returns:
    np.ndarray: A NumPy array where each row starts with the number and the remaining elements are from the list.
    """

    data = []
    
    # Strip newline characters
    line = line.strip()
    
    # Split the line into number and list parts
    number_part, list_part = line.split(' ', 1)
    
    # Convert the number to an integer
    number = int(number_part)
    
    # Safely convert the list string to an actual list using ast.literal_eval
    list_data = ast.literal_eval(list_part)
    
    # Combine the number and the list into one array and append to data
    data.append([number] + list_data)
    
    # Convert the list of lists into a NumPy array
    np_array = np.array(data)
    
    return np_array

