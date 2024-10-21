import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog



def main():
    
    # Load the video file
    video_path = './Converted.mov'
    cam = cv2.VideoCapture(video_path)

    # Check if the video was successfully opened
    if not cam.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    ret, im = cam.read()
    height, width = im.shape[:2]
    roi_pts = np.array([[300,height], [1300,350], [1600,300], [width,350], [width,height]])
    mask = np.zeros(im.shape[:2], dtype=np.uint8) 
    cv2.fillPoly(mask, [roi_pts], 255)
    

    while True:
        # Read the video frame by frame
        ret, im = cam.read()

        # Exit the loop if no more frames are available or the video ends
        if not ret:
            break

        masked_frame = cv2.bitwise_and(im, im, mask=mask)


        outputs = predictor(masked_frame)

        pred_boxes = outputs["instances"].pred_boxes
        boxes = pred_boxes.tensor.cpu().numpy()

        v = Visualizer(im, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        output_image = out.get_image()
        boundary_color = (0, 0, 255)  # Red color for the boundary
        cv2.polylines(output_image, [roi_pts], isClosed=True, color=boundary_color, thickness=3)


        cv2.imshow('OUTPUT', output_image)

        # Press 'q' to exit the video playback early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=='__main__':
    main()