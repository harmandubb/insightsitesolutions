import cv2
import numpy as np

def combine_close_boxes(boxes, distance_threshold=125):
    """
    Combines nearby bounding boxes into larger boxes.

    :param boxes: List of (x, y, w, h) bounding boxes.
    :param distance_threshold: The minimum distance between boxes to combine them.
    :return: List of combined bounding boxes.
    """
    if not boxes:
        return []

    # Convert boxes to an array for easier manipulation
    boxes_array = np.array(boxes)

    # Create a list to hold combined boxes
    combined_boxes = []

    while len(boxes_array) > 0:
        # Take the first box and start a new cluster
        x, y, w, h = boxes_array[0]
        current_cluster = [boxes_array[0]]
        boxes_array = np.delete(boxes_array, 0, axis=0)

        # Compare this box with all other boxes
        i = 0
        while i < len(boxes_array):
            # Check if the box is close enough to the current cluster
            x2, y2, w2, h2 = boxes_array[i]
            if (abs(x - x2) < distance_threshold and abs(y - y2) < distance_threshold):
                current_cluster.append(boxes_array[i])
                boxes_array = np.delete(boxes_array, i, axis=0)
            else:
                i += 1

        # Compute the minimum enclosing rectangle for the current cluster
        x_min = min([b[0] for b in current_cluster])
        y_min = min([b[1] for b in current_cluster])
        x_max = max([b[0] + b[2] for b in current_cluster])
        y_max = max([b[1] + b[3] for b in current_cluster])

        # Add the new combined box to the list
        combined_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return combined_boxes

def detect_vehicles_in_polygonal_roi(video_path, polygon_points):
    """
    Detects vehicles within a custom polygonal ROI in the video.

    :param video_path: Path to the video file.
    :param polygon_points: List of (x, y) tuples defining the polygonal ROI.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create a background subtractor
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        # Create a mask with the same size as the frame, and initialize it to black (zeros)
        mask = np.zeros(frame.shape[:2], dtype="uint8")

        # Convert polygon points to a numpy array of integers and reshape
        polygon_points = np.array(polygon_points, dtype=np.int32)
        polygon_points = polygon_points.reshape((-1, 1, 2))  # Ensure it's of shape (N, 1, 2)

        # Draw the filled polygon on the mask
        cv2.fillPoly(mask, [polygon_points], 255)

        # Apply the mask to the frame to get the ROI
        roi = cv2.bitwise_and(frame, frame, mask=mask)

        # Apply the background subtractor to the ROI
        foreground_mask = background_subtractor.apply(roi)

        # Perform some morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Store all detected bounding boxes
        boxes = []

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Only consider larger contours
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append((x, y, w, h))

        # Combine nearby bounding boxes
        combined_boxes = combine_close_boxes(boxes)

        # Draw the combined bounding boxes
        for (x, y, w, h) in combined_boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw the polygon on the frame for visualization
        cv2.polylines(frame, [polygon_points], isClosed=True, color=(0, 0, 255), thickness=2)

        # Display the full frame with bounding boxes and polygon ROI
        cv2.imshow('Vehicle Detection in Polygonal ROI', frame)

        # Press 'q' to exit the video playback
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

