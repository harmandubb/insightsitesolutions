def count_class_overtime(data_array, time_array, people_array, car_array, average_time_per_frame):
    time_array.append(data_array[0]*average_time_per_frame)
    people_array.append((data_array == 0).sum())
    car_array.append((data_array ==2).sum())

    return time_array, people_array, car_array

