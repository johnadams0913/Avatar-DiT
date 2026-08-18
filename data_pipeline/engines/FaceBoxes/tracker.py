import numpy as np
from .utils import KalmanBoxTracker, linear_assignment

class FaceBoxesTracker(object):
    def __init__(self, max_miss=1, min_length=150, min_size=128):
        self.trackers = []
        self.tracked_faces = []
        self.max_miss = max_miss
        self.min_size = min_size
        self.min_length = min_length

    def update(self, dets, frame_idx):
        dets = dets.copy()[:, :4]
        dets[:, 2:4] = dets[:, 2:4] + dets[:, :2]
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers),5))
        for t,trk in enumerate(self.trackers):
            pos = trk.predict()[0]
            trks[t,:] = [pos[0], pos[1], pos[2], pos[3], 0]

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)
        # update matched trackers with assigned detections
        for t,trk in enumerate(self.trackers):
            if(t not in unmatched_trks):
                d = matched[np.where(matched[:,1]==t)[0],0]
                trk.update(dets[d,:][0], frame_idx)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:], frame_idx) 
            self.trackers.append(trk)
        
        rmv_idx = len(self.trackers)
        for trk in reversed(self.trackers):
            time_since_update = frame_idx - trk.end_frame
            rmv_idx -= 1
            # remove dead tracklet
            if(time_since_update > self.max_miss):
                removed_trackers = self.trackers.pop(rmv_idx)
                if removed_trackers.end_frame - removed_trackers.start_frame > self.min_length:
                    self.tracked_faces.append(removed_trackers)

    def get_tracked_faces(self):
        for trk in self.trackers:
            if trk.end_frame - trk.start_frame > self.min_length:
                self.tracked_faces.append(trk)
        results = []
        for trk in self.tracked_faces:
            all_bbox = np.stack(trk.all_bbox)
            unified_bbox = [int(np.min(all_bbox[:, 0])), int(np.min(all_bbox[:, 1])), int(np.max(all_bbox[:, 2])), int(np.max(all_bbox[:, 3]))]
            avg_bbox = [int(np.mean(all_bbox[:, 0])), int(np.mean(all_bbox[:, 1])), int(np.mean(all_bbox[:, 2])), int(np.mean(all_bbox[:, 3]))]
            unified_bbox = [unified_bbox[0], unified_bbox[1], unified_bbox[2] - unified_bbox[0], unified_bbox[3] - unified_bbox[1]]
            avg_bbox = [avg_bbox[0], avg_bbox[1], avg_bbox[2] - avg_bbox[0], avg_bbox[3] - avg_bbox[1]]
            avg_face_size = 0.5 * (avg_bbox[2] + avg_bbox[3])
            area_ratio = (avg_bbox[2] * avg_bbox[3]) / (unified_bbox[2] * unified_bbox[3])
            if area_ratio < 0.25 or area_ratio > 2.0:
                continue # filter out small or large faces
            if avg_face_size < self.min_size:
                continue
            results.append({'start_frame': trk.start_frame, 'end_frame': trk.end_frame, 'bbox': unified_bbox})
        return results


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
    iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

    for d,det in enumerate(detections):
        for t,trk in enumerate(trackers):
            iou_matrix[d,t] = iou(det,trk)
    matched_indices = linear_assignment(-iou_matrix)

    unmatched_detections = []
    for d,det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t,trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(iou_matrix[m[0],m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def iou(bb_test,bb_gt):
    """
    Computes IUO between two bboxes in the form [x1,y1,x2,y2]
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
        + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)
