from file import read_file_by_lines
from analysis_parsing import extract_video_parameters, analysis_data_extract
from datetime import datetime, time
from analysis_data import count_class_overtime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATA_FILE = "video_data.txt"
VIDEO_PARAMETERS_FILE = "video_parameters.txt"
TIME_ANALYSIS_PER_PASS = 60  # seconds
DATA_SMOOTHING_WINDOW_CARS = 6
DATA_SMOOTHING_WINDOW_PEOPLE = 30

average_time_per_frame = 0


def main():
    time_array = []
    car_count_array = []
    people_count_array = []
    lines_to_read = True
    start_line = 0
    
    # Extract video parameters
    average_time_per_frame, date, start_time = extract_video_parameters(VIDEO_PARAMETERS_FILE)
    
    # Calculate the number of lines to read based on average time per frame
    num_lines_read = int(TIME_ANALYSIS_PER_PASS / average_time_per_frame)
    
    # Read the data
    while (lines_to_read):
        lines, lines_to_read = read_file_by_lines(DATA_FILE, num_lines_read, start_line=start_line*num_lines_read-1)
        for line in lines: 
            time_array, people_count_array, car_count_array = count_class_overtime(
                analysis_data_extract(line)[0], 
                time_array, 
                people_array=people_count_array, 
                car_array=car_count_array, 
                average_time_per_frame=average_time_per_frame
            )

        # Convert arrays to pandas Series
        time_pd = pd.Series(time_array)
        people_pd = pd.Series(people_count_array)
        car_pd = pd.Series(car_count_array)

        # Plot people count over time
        plt.figure(0)
        plt.scatter(time_pd, people_pd, label="People Count")
        plt.scatter(time_pd, people_pd.rolling(window=DATA_SMOOTHING_WINDOW_PEOPLE).mean(), label="Rolling Mean (People)")
        plt.plot(time_pd, np.round(people_pd.rolling(window=DATA_SMOOTHING_WINDOW_PEOPLE).mean()), label="Floored Rolling Mean (People)")
        plt.xlabel("Time")
        plt.ylabel("People Count")
        plt.title("People Count Over Time")
        plt.legend()

        # Plot car count over time
        plt.figure(1)
        plt.scatter(time_pd, car_pd, label="Car Count")
        plt.scatter(time_pd, car_pd.rolling(window=DATA_SMOOTHING_WINDOW_CARS).mean(), label="Rolling Mean (Car)")
        plt.plot(time_pd, np.floor(car_pd.rolling(window=DATA_SMOOTHING_WINDOW_CARS).mean()), label="Floored Rolling Mean (Car)")
        plt.xlabel("Time")
        plt.ylabel("Car Count")
        plt.title("Car Count Over Time")
        plt.legend()

        # Calculate differences for people and cars (change over time)
        diff_people = pd.Series(np.round(people_pd.rolling(window=DATA_SMOOTHING_WINDOW_PEOPLE).mean())).diff().dropna().astype(int)
        diff_cars = pd.Series(np.floor(car_pd.rolling(window=DATA_SMOOTHING_WINDOW_CARS).mean())).diff().dropna().astype(int)

        # Plot differences over time
        plt.figure(2)
        plt.scatter(time_pd[len(time_pd) - len(diff_people):], diff_people, label="People Change")
        plt.scatter(time_pd[len(time_pd) - len(diff_cars):], diff_cars, label="Car Change")
        plt.xlabel("Time")
        plt.ylabel("Change")
        plt.title("Change in People and Cars Over Time")
        plt.legend()

        # Summing positive and negative changes
        pos_diff_cars = diff_cars[diff_cars > 0].sum()
        neg_diff_cars = diff_cars[diff_cars < 0].sum()
        print("Cars entered the plaza (past minute):", pos_diff_cars)
        print("Cars left the plaza (past minute):", neg_diff_cars)

        pos_diff_people = diff_people[diff_people > 0].sum()
        neg_diff_people = diff_people[diff_people < 0].sum()
        print("People entered the plaza (past minute):", pos_diff_people)
        print("People left the plaza (past minute):", neg_diff_people)

        # Show plots
        plt.show()

        start_line = start_line + 1

if __name__ == "__main__":
    main()
