import cv2
import pytesseract

def extract_numbers_from_image(im, black_threshold=30, white_threshold=225):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cv2.imshow('GRAY TIME', gray)

    # Create a mask for black regions (pixel value 0)
    black_mask = cv2.inRange(gray, 0, black_threshold)  # Pixels that are exactly black
    cv2.imshow('BLACK MASK', black_mask)

    # Create a mask for white regions (pixel value 255)
    white_mask = cv2.inRange(gray, white_threshold, 255)  # Pixels that are exactly white
    cv2.imshow('WHITE MASK', white_mask)

    # Combine both masks using bitwise OR to get black and white regions
    combined_mask = cv2.bitwise_or(black_mask, white_mask)
    cv2.imshow('BLACK and WHITE MASK', combined_mask)

    # Apply the mask to the original image
    result = cv2.bitwise_and(im, im, mask=combined_mask)
    cv2.imshow('BLACK and WHITE IM', result)

    extracted_text = pytesseract.image_to_string(result)
    

    return extracted_text

def crop_to_time(im, top_left, bottom_right):
    x1, y1 = top_left
    x2, y2 = bottom_right

    cropped_im = im[y1:y2, x1:x2]

    return cropped_im
