import cv2
import pytesseract
import numpy as np
import re
from datetime import datetime

from llm import create_prompt, prompt_model

def extract_time_white_filter(im, black_threshold=20, white_threshold=220):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('GRAY TIME', gray)
    
    # Create a mask for white regions (pixel value 255)
    white_mask = cv2.inRange(gray, white_threshold, 255)  # Pixels that are exactly white
    # cv2.imshow('WHITE MASK', white_mask)

    denoised_w_img = cv2.medianBlur(white_mask, 3)
    cv2.imshow('WHITE MASK - NOISE FILTER', denoised_w_img)

    extracted_text = pytesseract.image_to_string(denoised_w_img)
    
    return extracted_text, denoised_w_img

def extract_time_black_filter(im, black_threshold=20, white_threshold=220):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('GRAY TIME', gray)
    
    # Create a mask for white regions (pixel value 255)
    black_mask = cv2.inRange(gray, 0, black_threshold)  # Pixels that are exactly white
    # cv2.imshow('BLACK MASK', black_mask)

    selective_noise_reduction = process_noisiest_section(black_mask)
    # cv2.imshow('BLACK MASK - SELEC NOISE', selective_noise_reduction)

    denoised_b_img = cv2.medianBlur(selective_noise_reduction, 3)
    cv2.imshow('BLACK MASK - SELEC NOISE - NOISE FILTER', denoised_b_img)

    extracted_text = pytesseract.image_to_string(denoised_b_img)
    
    return extracted_text, denoised_b_img

def extract_time_combined_filter(im_white, im_black):
    # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # # cv2.imshow('GRAY TIME', gray)

    # # Create a mask for white regions (pixel value 255)
    # white_mask = cv2.inRange(gray, white_threshold, 255)  # Pixels that are exactly white
    # # cv2.imshow('WHITE MASK', white_mask)

    # denoised_w_img = cv2.medianBlur(white_mask, 3)
    # # cv2.imshow('WHITE MASK - NOISE FILTER', denoised_w_img)
    
    # # Create a mask for white regions (pixel value 255)
    # black_mask = cv2.inRange(gray, 0, black_threshold)  # Pixels that are exactly white
    # # cv2.imshow('BLACK MASK', black_mask)

    # selective_noise_reduction = process_noisiest_section(black_mask)
    # # cv2.imshow('BLACK MASK - SELEC NOISE', selective_noise_reduction)

    # denoised_b_img = cv2.medianBlur(selective_noise_reduction, 3)
    # # cv2.imshow('BLACK MASK - SELEC NOISE - NOISE FILTER', denoised_b_img)

    # # Combine the denoised_w_img and erode_im using bitwise OR
    combined_img = cv2.bitwise_or(im_white, im_black)
    cv2.imshow('COMBINED IMAGE', combined_img)

    extracted_text = pytesseract.image_to_string(combined_img)
    
    return extracted_text

def calculate_noise_level(image_section):
    """Calculate noise level by counting edges (non-uniform pixels) in the section."""
    edges = cv2.Canny(image_section, 50, 150)
    noise_level = np.count_nonzero(edges)  # Non-zero pixels are considered noise
    return noise_level


def apply_noise_reduction(image_section):
    """Apply Gaussian blur as noise reduction to the noisy section."""
    kernel = np.ones((3,3),np.uint8) 
    return cv2.erode(image_section, kernel, iterations = 1)

def process_noisiest_section(image):
    # Get image dimensions
    height, width = image.shape

    # Split the image into top, bottom, left, and right halves
    top_half = image[:height // 2, :]
    bottom_half = image[height // 2:, :]
    left_half = image[:, :width // 2]
    right_half = image[:, width // 2:]

    # Calculate noise levels for each section
    noise_top = calculate_noise_level(top_half)
    noise_bottom = calculate_noise_level(bottom_half)
    noise_left = calculate_noise_level(left_half)
    noise_right = calculate_noise_level(right_half)

    # Find the section with the most noise
    noise_levels = {'top': noise_top, 'bottom': noise_bottom, 'left': noise_left, 'right': noise_right}
    most_noisy_section = max(noise_levels, key=noise_levels.get)

    # Apply noise reduction to the most noisy section
    if most_noisy_section == 'top':
        # print("Top half has the most noise. Applying noise reduction...")
        top_half_denoised = apply_noise_reduction(top_half)
        result = np.vstack((top_half_denoised, bottom_half))
    elif most_noisy_section == 'bottom':
        # print("Bottom half has the most noise. Applying noise reduction...")
        bottom_half_denoised = apply_noise_reduction(bottom_half)
        result = np.vstack((top_half, bottom_half_denoised))
    elif most_noisy_section == 'left':
        # print("Left half has the most noise. Applying noise reduction...")
        left_half_denoised = apply_noise_reduction(left_half)
        result = np.hstack((left_half_denoised, right_half))
    else:
        # print("Right half has the most noise. Applying noise reduction...")
        right_half_denoised = apply_noise_reduction(right_half)
        result = np.hstack((left_half, right_half_denoised))

    # Display the results
    # cv2.imshow('Original Image', image)
    # cv2.imshow('Processed Image', result)

    return result

def crop_to_time(im, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cropped_im = im[y1:y2, x1:x2]

    return cropped_im

def extract_valid_characters(input_string):
    # Extract valid characters (numerical, colon, and hyphen)
    cleaned_string = ''.join([char for char in input_string if char.isdigit() or char in [':', '-']])
    # Add a space after every four consecutive numbers
    return re.sub(r'(\d{4})(?=\d)', r'\1 ', cleaned_string)

def time_parser_from_llm(llm_response):
    lines = llm_response.strip().split('\n')

    # Patterns for date and time
    date_pattern = r"\d{2}-\d{2}-\d{4}"
    time_pattern = r"\d{2}:\d{2}:\d{2}"

    # Extract the date from the first line
    date_match = re.search(date_pattern, llm_response)
    date = date_match.group(0) if date_match else None

    # List to store the extracted times
    extracted_times = []

    # Process each line to extract the first time
    for line in lines:
        # Ensure the line starts with the index (e.g., '0.', '1.', etc.)
        if re.match(r"\d+\.", line):
            # Extract the first valid time from the line
            time_match = re.search(time_pattern, line)
            if time_match:
                extracted_times.append(time_match.group(0))  # Only the first time is captured

    # Output the extracted date and times
    print(f"Date: {date}")
    print("Times:")
    for time in extracted_times:
        print(time)

    return date, extracted_times

def time_difference_in_seconds(time_strings):
    # Parse the time strings into datetime objects
    time_format = "%H:%M:%S"
    times = [datetime.strptime(time, time_format) for time in time_strings]
    
    # Calculate the difference in seconds between adjacent times
    differences = []
    for i in range(1, len(times)):
        diff = (times[i] - times[i - 1]).total_seconds()
        differences.append(diff)
    
    return np.array(differences)

def outlier_present(array):
    mean = np.mean(array)
    std_dev = np.std(array)

    threshold = 1

    outliers = np.where(np.abs(array - mean) > threshold * std_dev)

    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Outliers at indices: {outliers}")

    if len(outliers[0]) > 0:
        return True
    else:
        return False
    

def get_video_time_parameters(time_data, TIME_FRAMES_RECORDED, TIME_COMPARISON_GROUP_SIZE,TIME_FRAME_FREQUENCY):
    time_difs = []
    average_time_per_frame = 0
    llm_response = ""

    prompt = create_prompt(time_data,TIME_FRAMES_RECORDED)
    llm_response = prompt_model(prompt)

    # conduct logic here to determine the time that has passed and the timer average per-frame
    # 1. parse the times to be in there seperate variables
    date, times = time_parser_from_llm(llm_response['response'])
    start_time = times[0]
    # 2. create 3 groups of 5
    for i in range(0, TIME_FRAMES_RECORDED // TIME_COMPARISON_GROUP_SIZE):
        time_dif = time_difference_in_seconds(times[i*TIME_COMPARISON_GROUP_SIZE:(i+1)*TIME_COMPARISON_GROUP_SIZE])
        print(time_dif)
        # 3. check if the sets are sequential if not then discared the set 
        if(np.all(time_dif > 0) and (len(time_dif) == TIME_COMPARISON_GROUP_SIZE-1)):
            outlier = outlier_present(time_dif)
            print(outlier)
            if not outlier:
                time_difs.append(time_dif)
    
    print(time_difs)
    if (len(time_difs)>0):
    # 4. The difference between the times shuold be fairly close. If more than 2 then there is an issue. 
    # 5. Find the average of the sets. 
        means_of_arrays = [np.mean(difs) for difs in time_difs]
        # 6. based on the above determine the best average rate perframe that should be adoppted. 
        average_time_per_frame = np.mean(means_of_arrays)/TIME_FRAME_FREQUENCY
        print(average_time_per_frame)
    
    return average_time_per_frame, date, start_time
