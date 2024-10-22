import cv2
import pytesseract
import numpy as np

def extract_numbers_from_image(im, black_threshold=20, white_threshold=225):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imshow('GRAY TIME', gray)

    # canny_im = cv2.Canny(gray, 100, 200)
    # cv2.imshow("CANNY", canny_im)

    # threshold_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv2.imshow('THRESHOLD', threshold_im)

    # kernel = np.ones((5,5),np.uint8)
    # erode_im = cv2.erode(gray, kernel, iterations = 1)
    # cv2.imshow('Erode', erode_im)

    # kernel = np.ones((5,5),np.uint8)
    # opening_im = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('OPENING', opening_im)

    # Create a mask for black regions (pixel value 0)
    black_mask = cv2.inRange(gray, 0, black_threshold)  # Pixels that are exactly black
    cv2.imshow('BLACK MASK', black_mask)

    denoised_img = cv2.medianBlur(black_mask, 5)
    cv2.imshow('BLACK MASK - NOISE FILTER', denoised_img)

    kernel = np.ones((2,2),np.uint8)
    erode_im = cv2.erode(denoised_img, kernel, iterations = 1)
    cv2.imshow('BLACK MASK - NOISE FILTER - ERODE', erode_im)

    # Create a mask for white regions (pixel value 255)
    white_mask = cv2.inRange(gray, white_threshold, 255)  # Pixels that are exactly white
    cv2.imshow('WHITE MASK', white_mask)

    denoised_img = cv2.medianBlur(white_mask, 3)
    cv2.imshow('NOISE FILTER', denoised_img)
    
    kernel = np.ones((3,3),np.uint8)
    dilate_im = cv2.dilate(denoised_img, kernel, iterations = 1)
    cv2.imshow('Dilate', dilate_im)

    # Check for shape mismatch and resize if necessary
    if erode_im.shape != dilate_im.shape:
        dilate_im = cv2.resize(dilate_im, (erode_im.shape[1], erode_im.shape[0]))

    # Check for dtype mismatch and convert if necessary
    if erode_im.dtype != dilate_im.dtype:
        dilate_im = dilate_im.astype(erode_im.dtype)

    # Perform bitwise operation
    combined_mask = cv2.bitwise_or(erode_im, dilate_im)
    cv2.imshow('RESULT', combined_mask)

    extracted_text = pytesseract.image_to_string(combined_mask)
    

    return extracted_text

def crop_to_time(im, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cropped_im = im[y1:y2, x1:x2]

    return cropped_im
