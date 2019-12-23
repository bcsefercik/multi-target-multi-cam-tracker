import os
import sys
import time
import json
import argparse

import numpy as np
import cv2
from timeit import time

from tracker import Tracker
from tracklet import TrackletManager

TRACKLET_LENGTH = 5
MIN_IOU = 0.7
MAX_COSINE_DISTANCE = 0.3

def main(opt, min_iou=MIN_IOU, max_cosine_distance=MAX_COSINE_DISTANCE, tracklet_length=TRACKLET_LENGTH):
    tracker = Tracker(min_iou=min_iou, max_cosine_distance=max_cosine_distance)
    tracklet_manager = TrackletManager(tracklet_length, tracker, os.path.join(opt.outputfolder, "tl{}_iou{}_d{}.txt".format(tracklet_length, min_iou, max_cosine_distance)))

    i = opt.start
    fps = 0.0
    while i <= opt.end:

        try:
            t1 = time.time()
            frame = cv2.imread(os.path.join(opt.framefolder, "{}.jpg".format(i))) if opt.show else None

            features_and_detections = {}

            features = None
            detections = None

            with open(os.path.join(opt.detectionfolder, "{}.json".format(i))) as f:
                detections = json.load(f)
            with open(os.path.join(opt.featurefolder, "{}.json".format(i))) as f:
                features = json.load(f)

            for f in features:
                features_and_detections[f["ID"]] = {
                    "features": np.array(f["features"], dtype=np.float32)
                }

            for d in detections:
                features_and_detections[int(d["ID"])]["bbox"] = np.array([d["xmin"], d["ymin"], d["xmax"], d["ymax"]], 
                                                                    dtype=np.int32)
            
            # print(features_and_detections)
            tracker.update(features_and_detections)
            tracklet_manager.update(features_and_detections)


            if opt.show:
                frame = cv2.imread(os.path.join(opt.framefolder, "{}.jpg".format(i)))
                for track in tracker.tracks:
                    bbox = track.last_bbox
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                    cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
                
                for detkey in features_and_detections:
                    bbox = features_and_detections[detkey]["bbox"]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,130,130), 2)
                    
                cv2.imshow('', frame)

            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            if i % 500 == 0:
                print("Frame ", i)
                print("FPS: %f"%(fps))

        except Exception as e:
            print("GG", e)
        
        i += 1
        # Press Q to stop!
        if opt.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--framefolder", type=str, default=".")
    argument_parser.add_argument("--detectionfolder", type=str, default=".")
    argument_parser.add_argument("--featurefolder", type=str, default=".")
    argument_parser.add_argument("--outputfolder", type=str, default=".")
    argument_parser.add_argument("--start", type=int)
    argument_parser.add_argument("--end", type=int)
    argument_parser.add_argument("--show", type=bool, default=False)
    opt = argument_parser.parse_args()

    for tracklet_length in [10, 5, 30]:
        for max_cosine_distance in [0.3, 0.1, 0.05]:
            for min_iou in [0.95, 0.75, 0.5]:
                main(opt, 
                    tracklet_length=tracklet_length,
                    min_iou=min_iou,
                    max_cosine_distance=max_cosine_distance)
