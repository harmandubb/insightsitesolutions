import cv2

from detector import Detector, filter_fields, visualize
from tracker import Tracker


def main():

    """
    detector used for object detection
    tracker used for tracking boxes
    counter used for checking whether a tracked object has crossed line and keeping counts
    aggregator used for checking time intervals (when to print out and reset counts)
    """
    detector = Detector()
    # tracker = Tracker()
    # counter = Counter(CROSS)
    # aggregator = Aggregator()
    
    # Load the video file
    video_path = 'Converted.mov'
    cam = cv2.VideoCapture(video_path)

    while True:
        # Read the video frame by frame
        ret, im = cam.read()

        # Exit the loop if no more frames are available or the video ends
        if not ret:
            break

        # Process the frame (e.g., display or any other operations)
        cv2.imshow("Frame", im)

        # Press 'q' to exit the video playback early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Detect objects on image
        outputs = detector.detect(im)
        fields = detector.get_fields(outputs)
        fields = filter_fields(fields)

        # Track the object boxes
        tracker.track(fields['pred_boxes'])
        
        # Check whether object crosses line
        # counter.check_crosses(tracker.objects)

        # # Visualize line and tracked objects on image
        # im_with_results = visualize_line(im,CROSS)
        # im_with_results = visualize_tracker(im_with_results,tracker)
        
        # Get counts and display that on image
        # results = counter.get_results()
        # im_with_results = display_text_box(im_with_results,f"{results}")

        # Print results to file and reset counts
        # if aggregator.check():
        #     counter.print_results()
        #     counter.reset()

        # Press 'q' to exit the video playback early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=='__main__':
    main()