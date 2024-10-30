from file import read_file_by_lines
from analysis_parsing import extract_video_parameters, analysis_data_extract
from datetime import datetime, time
from analysis_data import count_class_overtime, object_change_over_time, object_change_in_rolling_average
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time_functions import TimeTracker

DATA_FILE = "videos\Elginpark_Cam4_Elgin_Park_Elgin_Park_20240918113606_20240918115914_12838717.mp4_video_data.txt"
VIDEO_PARAMETERS_FILE = "videos\Elginpark_Cam4_Elgin_Park_Elgin_Park_20240918113606_20240918115914_12838717.mp4_video_parameters.txt"
TIME_ANALYSIS_PER_PASS = 10*60  # seconds
DATA_SMOOTHING_WINDOW_CARS = 3
DATA_SMOOTHING_WINDOW_PEOPLE = 10

average_time_per_frame = 0


def main():
    time_array = []
    car_count_array = []
    people_count_array = []
    read_lines = True
    num_lines_read = 1
    absolute_datetime_array = [] #tracking the time inputs over time
    
    # Extract video parameters
    average_time_per_frame, date, start_time = extract_video_parameters(VIDEO_PARAMETERS_FILE)
    time_tracker = TimeTracker(start_time, date)
    time_tracker.set_time_per_frame(average_time_per_frame)

    # Calculate the number of lines to read based on average time per frame
    print("TIME ANALYSIS PER PASS:", TIME_ANALYSIS_PER_PASS)
    print("Average time per frame:", average_time_per_frame)
    num_lines_to_read = int(TIME_ANALYSIS_PER_PASS / average_time_per_frame)
    
    

    # Read the data
    while (read_lines):
        print("Lines to read:", num_lines_to_read)
        lines, read_lines = read_file_by_lines(DATA_FILE, num_lines_to_read, start_line=num_lines_read)
        print("LINES READ:", len(lines))
        print("LINES:", lines)
        num_lines_read += len(lines)
        for line in lines: 
            # print("LINE:", line)
            # could modify this function to modify the time using timedelta(secounds=n)
            people_count_array, car_count_array = count_class_overtime(
                analysis_data_extract(line)[0], 
                time_array, 
                people_array=people_count_array, 
                car_array=car_count_array, 
                average_time_per_frame=average_time_per_frame
            )
            absolute_datetime_array.append(time_tracker.current_datetime)
            time_tracker.increment_time(1)

        time_pd = pd.Series(absolute_datetime_array)
        people_pd = pd.Series(people_count_array)
        car_pd = pd.Series(car_count_array)

        rolling_average_people, rolling_average_cars = object_change_over_time(time_pd=time_pd,people_pd=people_pd,car_pd=car_pd,data_smoothing_window_cars=DATA_SMOOTHING_WINDOW_CARS, data_smoothing_window_people=DATA_SMOOTHING_WINDOW_PEOPLE, show_plots=True)

        pos_diff_cars, neg_diff_cars, pos_diff_people, neg_diff_people = object_change_in_rolling_average(rolling_average_people, rolling_average_cars,time_pd, show_plots=True)

        # metrics to focus on: rolling average people, rolling average cars, pos diff caras and neg diff cars

        plt.show()


if __name__ == "__main__":
    main()
