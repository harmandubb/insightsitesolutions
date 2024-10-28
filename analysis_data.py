import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np 
import pandas as pd

def count_class_overtime(data_array, time_array, people_array, car_array, average_time_per_frame):
    time_array.append(data_array[0]*average_time_per_frame)
    people_array.append((data_array == 0).sum())
    car_array.append((data_array ==2).sum())

    return time_array, people_array, car_array

def object_change_over_time(time_pd, people_pd, car_pd, data_smoothing_window_people, data_smoothing_window_cars, show_plots=False):
    # Plot people count over time

    # Calculate differences for people and cars (change over time)
    rolling_average_people = pd.Series(np.round(people_pd.rolling(window=data_smoothing_window_people).mean()))
    
    rolling_average_cars = pd.Series(np.round(car_pd.rolling(window=data_smoothing_window_cars).mean())).diff().dropna().astype(int)
    

    if(show_plots):
        plt.figure(0)
        # plt.scatter(time_pd, people_pd, label="People Count")
        plt.scatter(time_pd, people_pd.rolling(window=data_smoothing_window_people).mean(), label="Rolling Mean (People)")
        plt.scatter(time_pd, np.round(people_pd.rolling(window=data_smoothing_window_people).mean()), label="Rounded Rolling Mean (People)")

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Show hours:minutes
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=1800))  # Tick every 10 minutes

        # Rotate the x-axis labels for better readability
        plt.gcf().autofmt_xdate()
        plt.xlabel("Time")
        plt.ylabel("People Count")
        plt.title("People Count Over Time")
        plt.legend()

        # Plot car count over time
        plt.figure(1)
        # plt.scatter(time_pd, car_pd, label="Car Count")
        plt.scatter(time_pd, car_pd.rolling(window=data_smoothing_window_cars).mean(), label="Rolling Mean (Car)")
        plt.scatter(time_pd, np.round(car_pd.rolling(window=data_smoothing_window_cars).mean()), label="Rounded Rolling Mean (Car)")

        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Show hours:minutes
        plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=1800))  # Tick every 10 minutes
        
        plt.xlabel("Time")
        plt.ylabel("Car Count")
        plt.title("Car Count Over Time")
        plt.legend()
        plt.show()

    return rolling_average_people, rolling_average_cars

    
    

def object_change_in_rolling_average(rolling_average_people, rolling_average_cars, time_pd, show_plots=False):
    diff_people = rolling_average_people.diff().dropna().astype(int)
    diff_cars = rolling_average_cars.diff().dropna().astype(int)

    if (show_plots):
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

    if(show_plots):
        print("Cars entered the plaza (past minute):", pos_diff_cars)
        print("Cars left the plaza (past minute):", neg_diff_cars)

    pos_diff_people = diff_people[diff_people > 0].sum()
    neg_diff_people = diff_people[diff_people < 0].sum()
    if(show_plots):
        print("People entered the plaza (past minute):", pos_diff_people)
        print("People left the plaza (past minute):", neg_diff_people)

        # Show plots
        plt.show()

    return pos_diff_cars, neg_diff_cars, pos_diff_people, neg_diff_people
