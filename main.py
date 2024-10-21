import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog


def main():
    
    # Load the video file
    video_path = 'Converted.mov'
    cam = cv2.VideoCapture(video_path)

    # cfg = get_cnf()
    

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

if __name__=='__main__':
    main()