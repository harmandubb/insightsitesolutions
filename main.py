import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
import pytesseract
from extract_time import crop_to_time, get_time_filters, extract_valid_characters, get_video_time_parameters, extarct_time_experimental
from file import write_to_file

from tracker import match_boxes

from torchvision.ops import nms

from llm import prompt_model, create_prompt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from pathlib import Path

# Constants
TIME_FRAMES_RECORDED=15
TIME_FRAME_FREQUENCY=4
TOTAL_TIME_FRAMES=TIME_FRAMES_RECORDED*TIME_FRAME_FREQUENCY
TIME_COMPARISON_GROUP_SIZE=2
VIDEO_DATA_FILE_SUFFIX="video_data.txt"
FRAME_SKIP=10


def main():
    frame_counter = 0; 
    
    # Load the video file
    # video_folder_path = Path("videos")
    video_folder_path = Path("videos")

    # video_folder_path = './Converted.mov'

    video_extensions = ['.mp4', '.mov']  # Add more extensions if needed

    for video_file in video_folder_path.iterdir():
        if video_file.suffix.lower() in video_extensions:
            print(f"Processing file: {video_file}")

            cam = cv2.VideoCapture(video_file)

            # Check if the video was successfully opened
            if not cam.isOpened():
                print(f"Error: Could not open video {video_file}")
                return

            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.FILTER_CLASS = [2,0,7]
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
            
            average_time_per_frame = 0

            video_tracking = []
            class_pairs = []

            # flags 
            do_llm_time_prompt = True

            write_to_file(str(video_file) + "_" + VIDEO_DATA_FILE_SUFFIX, "", mode="w")
        else: 
            return


        while True:
            # Read the video frame by frame
            ret, im = cam.read()
            
            # Exit the loop if no more frames are available or the video ends
            if not ret:
                break
            if (frame_counter == FRAME_SKIP):
                
                if (do_llm_time_prompt):
                    # still need to do the time llm thing
                    if ((frame_iterator % TIME_FRAME_FREQUENCY) == 0):
                        time_crop = crop_to_time(im, (200, 950), (800, height))
                        extarct_time_experimental(time_crop)
                        white_filter, _, combined_filter = get_time_filters(time_crop)
                        white_filter_time = extract_valid_characters(white_filter)
                        combined_filter_time = extract_valid_characters(combined_filter)
                        if(len(combined_filter_time)>0):
                            time_data.append([time_frames_count, white_filter_time, combined_filter_time])
                            time_frames_count = time_frames_count + 1

                    

                    if ((time_frames_count == TIME_FRAMES_RECORDED)):
                        average_time_per_frame, date, start_time = get_video_time_parameters(time_data, TIME_FRAMES_RECORDED, TIME_COMPARISON_GROUP_SIZE,TIME_FRAME_FREQUENCY)
                        print("AVERAGE_TIME_PER_FRAME:", average_time_per_frame)
                        print("DATE:", date)
                        print("START_TIME", start_time)
                        write_to_file(str(video_file) + "_video_parameters.txt", [
                                                                f"Average_TIME_PER_FRAME: {average_time_per_frame}",
                                                                f"DATE: {date}",
                                                                f"START_TIME: {start_time}"
                                                            ],
                                        mode='w')

                        if average_time_per_frame >  0:
                            do_llm_time_prompt = False
                        else:
                            do_llm_time_prompt = True
                            time_frames_count = 0
                            frame_iterator = 0

                frame_iterator = frame_iterator + 1

                masked_frame = cv2.bitwise_and(im, im, mask=mask)
                outputs = predictor(masked_frame)

                pred_boxes = outputs["instances"].pred_boxes.tensor
                pred_classes = outputs["instances"].pred_classes
                scores = outputs["instances"].scores

                curr_keep = nms(pred_boxes, scores,0.5)


                if len(prev_frame_boxes) > 0:
                    matched_pairs, class_pairs = match_boxes(prev_frame_boxes,pred_boxes,prev_frame_classes, pred_classes, prev_keep, curr_keep)
                    # print("Matched boxes between frames:", matched_pairs)
                    # print("Matches classes between frames:", class_pairs)


                video_tracking.append([frame_iterator, class_pairs])

                write_to_file(str(video_file) + "_" + VIDEO_DATA_FILE_SUFFIX,[frame_iterator, class_pairs], mode='a')

                # print(video_tracking[frame_iterator-1])    

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

                frame_counter = 0

            else: 
                frame_counter += 1

            # Press 'q' to exit the video playback early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return 
if __name__=='__main__':
    main()