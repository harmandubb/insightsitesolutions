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

def match_boxes(previous_frame_boxes, current_frame_boxes, previous_class, current_class, prev_keep, curr_keep, iou_threshold=0.5):
    matched_pairs = []
    class_pairs = []

    for current_idx in curr_keep:
        best_iou = 0
        best_prev_box = None
        best_prev_idx = None

        for prev_idx in prev_keep:
            iou = compute_iou(current_frame_boxes[current_idx], previous_frame_boxes[prev_idx])
            if iou > best_iou:
                # Consider it as the same object
                best_iou = iou
                best_prev_box = previous_frame_boxes[prev_idx]
                best_prev_idx = prev_idx

        if best_iou > iou_threshold:
            matched_pairs.append((best_prev_box,current_frame_boxes[current_idx]))
            # check if the labels are the same 
            if (((current_class[current_idx] == 2) or (current_class[current_idx] == 7)) and ((previous_class[best_prev_idx] == 2) or (previous_class[best_prev_idx] == 7))):
                class_pairs.append(2)
            elif (current_class[current_idx] == previous_class[best_prev_idx]): 
                class_pairs.append(current_class[current_idx].item())
    
    return matched_pairs, class_pairs
