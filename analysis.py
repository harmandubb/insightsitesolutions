from file import read_file_by_lines
from analysis_parsing import extract_video_parameters, analysis_data_extract
from datetime import datetime, time
from analysis_data import count_class_overtime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

DATA_FILE = "video_data.txt"
VIDEO_PARAMETERS_FILE = "video_parameters.txt"
TIME_ANALYSIS_PER_PASS = 60 #secounds

average_time_per_frame = 0


def main():
    time_array = []
    car_count_array = []
    people_count_array = []
    average_time_per_frame, date, start_time = extract_video_parameters(VIDEO_PARAMETERS_FILE)
    
    num_lines_read = math.floor(TIME_ANALYSIS_PER_PASS/average_time_per_frame)

    lines = read_file_by_lines(DATA_FILE, num_lines_read)
    for line in lines: 
        time_array, people_count_array, car_count_array = count_class_overtime(analysis_data_extract(line)[0], time_array, 
                                                                               people_array=people_count_array, 
                                                                               car_array=car_count_array, 
                                                                               average_time_per_frame=average_time_per_frame)
        print("TIME:", time_array)
        print("People:", people_count_array)
        print("Car:", car_count_array)

        time_pd = pd.Series(time_array)
        people_pd = pd.Series(people_count_array)
        car_pd = pd.Series(car_count_array)

    plt.figure(0)
    plt.scatter(time_pd, people_pd)
    plt.scatter(time_pd, people_pd.rolling(window=3).mean())
    plt.scatter(time_pd, np.floor(people_pd.rolling(window=3).mean()))
    cleaned_people = np.floor(people_pd.rolling(window=3).mean())
    plt.xlabel("Time")
    plt.ylabel("People Count")
    plt.title("People Count Over Time")

    plt.figure(1)
    plt.scatter(time_pd, car_pd)
    plt.scatter(time_pd, car_pd.rolling(window=3).mean())
    cleaned_cars = np.floor(car_pd.rolling(window=3).mean())
    plt.scatter(time_pd, np.floor(car_pd.rolling(window=3).mean()))
    plt.xlabel("Time")
    plt.ylabel("Car Count")
    plt.title("Car Count Over Time")

    diff_people = pd.Series(cleaned_people).diff().dropna().astype(int)
    diff_cars = pd.Series(cleaned_cars).diff().dropna().astype(int)

    plt.figure(2)
    print(len(time_pd))
    print(len(diff_people))
    plt.scatter(time_pd[:-3], diff_people)
    plt.scatter(time_pd[:-3], diff_cars)

    pos_diff_cars = diff_cars[diff_cars>0].sum()
    neg_diff_cars = diff_cars[diff_cars<0].sum()

    print("Cars entered the plaza: (past minute):", pos_diff_cars)
    print("Cars left the plaza: (past minute):", neg_diff_cars)


    pos_diff_people = diff_people[diff_people>0].sum()
    neg_diff_people = diff_people[diff_people<0].sum()

    print("People entered the plaza: (past minute):", pos_diff_people)
    print("People left the plaza: (past minute):", neg_diff_people)



    
    plt.show()


if __name__ == "__main__":
    main()