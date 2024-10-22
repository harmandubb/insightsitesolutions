import cv2
import pytesseract
import numpy as np

def extract_numbers_from_image(im, black_threshold=20, white_threshold=220):
    smoothed_image = cv2.blur(im, (2, 2))
    cv2.imshow('SMOOTH', smoothed_image)

    gray = cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('GRAY TIME', gray)

    # Create a mask for black regions (pixel value 0)
    black_mask = cv2.inRange(gray, 0, black_threshold)  # Pixels that are exactly black
    cv2.imshow('BLACK MASK', black_mask)

    kernel = np.ones((2,2),np.uint8)
    erode_im = cv2.erode(black_mask, kernel, iterations = 1)
    cv2.imshow('BLACK MASK - ERODE', erode_im)

    denoised_b_img = cv2.medianBlur(erode_im, 3)
    cv2.imshow('BLACK MASK - ERODE - NOISE FILTER', denoised_b_img)
    
    # Create a mask for white regions (pixel value 255)
    white_mask = cv2.inRange(gray, white_threshold, 255)  # Pixels that are exactly white
    cv2.imshow('WHITE MASK', white_mask)

    denoised_w_img = cv2.medianBlur(white_mask, 3)
    cv2.imshow('WHITE MASK - NOISE FILTER', denoised_w_img)

    extracted_text = pytesseract.image_to_string(denoised_w_img)
    
    return extracted_text

def crop_to_time(im, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cropped_im = im[y1:y2, x1:x2]

    return cropped_im
