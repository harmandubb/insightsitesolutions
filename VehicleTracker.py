import cv2
import numpy as np

def is_line_crossed(centroid, line_start, line_end):
    """
    Check if the centroid crosses a finite line segment defined by line_start and line_end.

    :param centroid: The (x, y) coordinates of the vehicle's centroid.
    :param line_start: The (x1, y1) coordinates of the start of the line.
    :param line_end: The (x2, y2) coordinates of the end of the line.
    :return: True if the centroid crosses the line, False otherwise.
    """
    # Vector from line_start to line_end
    line_vec = np.array(line_end) - np.array(line_start)
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    # Vector from line_start to the centroid
    point_vec = np.array(centroid) - np.array(line_start)

    # Project point_vec onto the line_vec to find the projection scalar
    projection_length = np.dot(point_vec, line_vec_norm)
    
    # Check if the projection scalar falls within the line segment
    if 0 <= projection_length <= np.linalg.norm(line_vec):
        # Check if the centroid is close enough to the line (within some small threshold)
        perpendicular_dist = np.linalg.norm(point_vec - projection_length * line_vec_norm)
        if perpendicular_dist < 10:  # Adjust threshold as needed
            return True
    return False

def detect_motion_and_track_color(video_path, line_start, line_end, tracking_polygon):
    """
    Detects motion in the video, tracks a vehicle crossing a finite line, and determines the color of the vehicle.

    :param video_path: Path to the video file.
    :param line_start: The (x1, y1) coordinates of the start of the line to check for crossings.
    :param line_end: The (x2, y2) coordinates of the end of the line to check for crossings.
    :param tracking_polygon: A list of (x, y) tuples defining the vertices of the polygon for vehicle tracking.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Create background subtractor for motion detection
    background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50)

    # Initialize tracking flag and color storage
    tracking = False
    tracked_vehicle_color = None

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        # Resize the frame (optional, depending on input video size)
        # frame = cv2.resize(frame, (640, 480))

        # Apply the background subtractor for motion detection
        fg_mask = background_subtractor.apply(frame)

        # Perform some morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the motion mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the finite line across which motion will be detected
        cv2.line(frame, line_start, line_end, (0, 0, 255), 2)

        # Draw the polygon on the frame for visualization
        cv2.polylines(frame, [np.array(tracking_polygon, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Variable to hold vehicle bounding box
        vehicle_box = None

        for contour in contours:
            # Filter small contours (motion noise)
            if cv2.contourArea(contour) < 500:
                continue

            # Get the bounding box for the detected motion
            x, y, w, h = cv2.boundingRect(contour)
            vehicle_box = (x, y, w, h)

            # Get the centroid of the vehicle's bounding box
            centroid = (x + w // 2, y + h // 2)

            # Check if the centroid crosses the finite line
            if is_line_crossed(centroid, line_start, line_end):
                tracking = True  # Start tracking the vehicle
                cv2.putText(frame, "Motion Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # If motion was detected and tracking is active
        if tracking and vehicle_box is not None:
            x, y, w, h = vehicle_box

            # Draw the bounding box around the vehicle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if the vehicle's centroid is inside the polygon
            is_in_polygon = cv2.pointPolygonTest(np.array(tracking_polygon, np.int32), centroid, False)

            if is_in_polygon >= 0:  # If the centroid is inside the polygon
                # Crop the vehicle area from the frame
                vehicle_roi = frame[y:y + h, x:x + w]

                # Determine the dominant color of the vehicle (average color in the bounding box)
                avg_color_per_row = np.average(vehicle_roi, axis=0)
                avg_color = np.average(avg_color_per_row, axis=0)

                # Convert the color to integer (for displaying)
                vehicle_color = tuple(map(int, avg_color))

                # Store the detected vehicle color
                if tracked_vehicle_color is None:
                    tracked_vehicle_color = vehicle_color
                    print(f"Vehicle color detected: {tracked_vehicle_color}")

                # Display the detected color
                cv2.putText(frame, f"Color: {tracked_vehicle_color}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracked_vehicle_color, 2)

        # Display the frame with motion and vehicle tracking
        cv2.imshow('Motion Detection and Vehicle Tracking', frame)

        # Press 'q' to exit the video playback
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()