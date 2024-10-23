import cv2
import pytesseract
import numpy as np

def extract_time_white_filter(im, black_threshold=20, white_threshold=220):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('GRAY TIME', gray)
    
    # Create a mask for white regions (pixel value 255)
    white_mask = cv2.inRange(gray, white_threshold, 255)  # Pixels that are exactly white
    # cv2.imshow('WHITE MASK', white_mask)

    denoised_w_img = cv2.medianBlur(white_mask, 3)
    cv2.imshow('WHITE MASK - NOISE FILTER', denoised_w_img)

    extracted_text = pytesseract.image_to_string(denoised_w_img)
    
    return extracted_text

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
    
    return extracted_text

def extract_time_combined_filter(im, black_threshold=20, white_threshold=220):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('GRAY TIME', gray)

    # Create a mask for white regions (pixel value 255)
    white_mask = cv2.inRange(gray, white_threshold, 255)  # Pixels that are exactly white
    # cv2.imshow('WHITE MASK', white_mask)

    denoised_w_img = cv2.medianBlur(white_mask, 3)
    # cv2.imshow('WHITE MASK - NOISE FILTER', denoised_w_img)
    
    # Create a mask for white regions (pixel value 255)
    black_mask = cv2.inRange(gray, 0, black_threshold)  # Pixels that are exactly white
    # cv2.imshow('BLACK MASK', black_mask)

    selective_noise_reduction = process_noisiest_section(black_mask)
    # cv2.imshow('BLACK MASK - SELEC NOISE', selective_noise_reduction)

    denoised_b_img = cv2.medianBlur(selective_noise_reduction, 3)
    # cv2.imshow('BLACK MASK - SELEC NOISE - NOISE FILTER', denoised_b_img)

    # Combine the denoised_w_img and erode_im using bitwise OR
    combined_img = cv2.bitwise_or(denoised_w_img, denoised_b_img)
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

