import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
import pytesseract
from extract_time import crop_to_time, extract_time_white_filter, extract_time_combined_filter, extract_time_black_filter, extract_valid_characters, time_parser_from_llm, time_difference_in_seconds, outlier_present

from tracker import match_boxes

from torchvision.ops import nms

from llm import prompt_model, create_prompt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Constants
TIME_FRAMES_RECORDED=15
TIME_FRAME_FREQUENCY=4
TOTAL_TIME_FRAMES=TIME_FRAMES_RECORDED*TIME_FRAME_FREQUENCY
TIME_COMPARISON_GROUP_SIZE=5


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
     
    prev_frame_boxes = []
    prev_frame_classes = []
    prev_keep = []
    time_data = []
    time_frames_count = 0
    frame_iterator = 0
    llm_response = ""
    time_difs = []
    average_time_per_frame = 0

    # flags 
    do_llm_time_prompt = True

    while True:
        # Read the video frame by frame
        ret, im = cam.read()

        # Exit the loop if no more frames are available or the video ends
        if not ret:
            break

        
        if (do_llm_time_prompt):
            # still need to do the time llm thing
            if ((frame_iterator % TIME_FRAME_FREQUENCY) == 0):
                # record a time frame to extract time from
                time_crop = crop_to_time(im, (200, 950), (800, height))
                time_white, filtered_white = extract_time_white_filter(time_crop)
                # print("WHITE FILTER", time_white)
                time_black, filtered_black = extract_time_black_filter(time_crop)
                # print("BLACK FILTER", time_black)
                time_combined = extract_time_combined_filter(filtered_white,filtered_black)
                # print("COMBINE FILTER", time_combined)

                time_data.append([time_frames_count, extract_valid_characters(time_white), extract_valid_characters(time_combined)])
                time_frames_count = time_frames_count + 1

                masked_frame = cv2.bitwise_and(im, im, mask=mask)
                
            frame_iterator = frame_iterator + 1

            if ((time_frames_count == TIME_FRAMES_RECORDED)):
                prompt = create_prompt(time_data,TIME_FRAMES_RECORDED)
                llm_response = prompt_model(prompt)
                do_llm_time_prompt = False

                # conduct logic here to determine the time that has passed and the timer average per-frame
                # 1. parse the times to be in there seperate variables
                date, times = time_parser_from_llm(llm_response['response'])
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
                    do_llm_time_prompt = False
                else:
                    do_llm_time_prompt = True
                    time_frames_count = 0
                    frame_iterator = 0



        outputs = predictor(masked_frame)

        pred_boxes = outputs["instances"].pred_boxes.tensor
        pred_classes = outputs["instances"].pred_classes
        scores = outputs["instances"].scores

        curr_keep = nms(pred_boxes, scores,0.5)


        if len(prev_frame_boxes) > 0:
            matched_pairs, class_pairs = match_boxes(prev_frame_boxes,pred_boxes,prev_frame_classes, pred_classes, prev_keep, curr_keep)
            # print("Matched boxes between frames:", matched_pairs)
            # print("Matches classes between frames:", class_pairs)

        prev_frame_boxes = pred_boxes
        prev_frame_classes = pred_classes
        prev_keep = curr_keep

        v = Visualizer(im, MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        output_image = out.get_image()
        boundary_color = (0, 0, 255)  # Red color for the boundary
        cv2.polylines(output_image, [roi_pts], isClosed=True, color=boundary_color, thickness=3)

        scale_percent = 50
        viewport_width = int(width * scale_percent / 100)
        viewport_height = int(height * scale_percent / 100)
        frame_resize = cv2.resize(output_image, (viewport_width, viewport_height))
        
        cv2.imshow('OUTPUT', frame_resize)

        # Press 'q' to exit the video playback early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=='__main__':
    main()