"""
As implemented in https://github.com/abewley/sort but with some modifications

For each detected item, it computes the intersection over union (IOU) w.r.t. each tracked object. (IOU matrix)
Then, it applies the Hungarian algorithm (via linear_assignment) to assign each det. item to the best possible
tracked item (i.e. to the one with max. IOU).

Note: a more recent approach uses a Deep Association Metric instead.
see https://github.com/nwojke/deep_sort
"""

import numpy as np
import lap
import math

def associate_detections_to_trackers(detections, trackers, iou_threshold = 40):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2),dtype=int), np.arange(len(detections)), np.empty((0, 5),dtype=int)
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d,t] = line_iou(det, trk)

            
    # Minimise the total assignment cost
    matched_indices = linear_assignment(iou_matrix)
        
    
    # Find unmatched detections
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    # Find lost tracklets
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] > iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

def line_iou(line_det, line_trk):
    
    """
    Computes the matching between two lines [rho1, theta1], [rho2, theta2]
    """
    y_list = np.array(list(range(295,595,5)))
    
    a_det, b_det = polar2cart(line_det[0], line_det[1])
    a_trk, b_trk = polar2cart(line_trk[0], line_trk[1])
    
    x_det = (y_list - b_det)/a_det
    x_trk = (y_list - b_trk)/a_trk
    
    abs_sum = np.sum(abs(x_det - x_trk))
    
    return(abs_sum/(295))

def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i], i] for i in x if i >= 0])

def polar2cart(rho, theta):
    a = -math.tan(theta)
    b = rho* math.sqrt(1 + a**2)
    return a,b