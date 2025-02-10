import requests
import json

import ollama

def prompt_model(user_prompt, model_use='deepseek-r1:8b'):

    response = ollama.generate(model=model_use, prompt=user_prompt)
    
    print(response['response'])

    return response


def create_prompt(time_lists, num_entries):
    # Ensure num_entries doesn't exceed the length of the time_lists
    if num_entries > len(time_lists):
        raise ValueError("num_entries exceeds the length of time_lists.")
    
    # Build the data input string
    data_inputs = "[ " + ", ".join([f"[ '{str(time_lists[i][0])}', '{str(time_lists[i][1])}', '{str(time_lists[i][2])}' ]" 
                                    for i in range(num_entries)]) + " ]"
    
    # Define the starting and ending parts of the prompt
    startprompt = (
        "Every sublist in the below collection of data is organized as follows [Index Number, Reading 1, Reading 2]. "
        "Reading 1 and Reading 2 represent a date and time, but there are imperfections in the data. I want you to look "
        "at readings 1 and 2 and produce the date and time in this format: MM-DD-YYYY HH:MM:SS where MM represents the month "
        "in numerical value, DD represents the day in numerical value, YYYY represents the year, HH represents the hour in the day, "
        "MM represents the minutes, and SS represents the seconds. For example, if the input is: [0, '09-18-2024 9', '09-18-2024 09:34:19'], "
        "the output should be: 09-18-2024 09:34:19. Another example is if the input is: [27, '09-18-2024-09:35:49', '09-18-2024-09:35:49'], "
        "then the output should be: 09-18-2024 09:35:49. Below is the data:"
    )
    
    # Dynamically insert num_entries into the end prompt
    endprompt = (
        f"Look at each sublist and output a response in this format: index. MM-DD-YYYY HH:MM:SS for each group of inputs that are provided. If you suspect an error is present, "
        f"use the previous data to make an educated guess. The output should have {num_entries} entries (one for each sublist index). "
        "The data sets were taken sequentially, so the entry at index 1 would have a time before the entry at index 2. If this pattern is not mentained for a certain entry when replace that entery with the previous time value"
    )

    # Combine the parts into the final prompt
    prompt = f"{startprompt} {data_inputs} {endprompt}"

    return prompt
