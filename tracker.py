def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    # Compute intersection
    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    # Compute union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

def match_boxes(previous_frame_boxes, current_frame_boxes, previous_class, current_class, iou_threshold=0.5):
    matched_pairs = []
    class_pairs = []

    for current_box in current_frame_boxes:
        best_iou = 0
        best_prev_box = None

        for prev_box in previous_frame_boxes:
            iou = compute_iou(current_box, prev_box)
            if iou > best_iou:
                # Consider it as the same object
                best_iou = iou
                best_prev_box = prev_box

        if best_iou > iou_threshold:
            matched_pairs.append((best_prev_box,current_box))
            # check if the labels are the same 
            if (((current_class == 2) or (current_class == 7)) and (previous_class == 2) or (previous_class == 7)):
                class_pairs.append(2)
            elif (current_class == previous_class): 
                class_pairs.append(current_class)
    
    return matched_pairs, 
