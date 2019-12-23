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

TRACKLET_LENGTH = 10
MIN_IOU = 0.7
MAX_COSINE_DISTANCE = 0.3

def main(opt, min_iou=MIN_IOU, max_cosine_distance=MAX_COSINE_DISTANCE, tracklet_length=TRACKLET_LENGTH):
    if opt.write_video:
    # Define the codec and create VideoWriter object
        w = int(1920)
        h = int(1080)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("output-tl{}_iou{}_d{}.mp4".format(tracklet_length, min_iou, max_cosine_distance), fourcc, 30, (w, h))
        # list_file = open('detection.txt', 'w')
        frame_index = -1 

    tracker = Tracker(min_iou=min_iou, max_cosine_distance=max_cosine_distance)
    tracklet_manager = TrackletManager(tracklet_length, tracker, os.path.join(opt.outputfolder, "tl{}_iou{}_d{}.txt".format(tracklet_length, min_iou, max_cosine_distance)))

    i = opt.start
    fps = 0.0
    while i <= opt.end:
        try:
            t1 = time.time()

            features_and_detections = {}
            features = {}
            detections = {}

            try:
                with open(os.path.join(opt.detectionfolder, "{}.json".format(i))) as f:
                    detections = json.load(f)
            except FileNotFoundError:
                print("INFO: No detections for frame", i)
                # raise FileNotFoundError
            
            try:
                with open(os.path.join(opt.featurefolder, "{}.json".format(i))) as f:
                    features = json.load(f)
            except FileNotFoundError:
                print("INFO: No features for frame", i)
                # raise FileNotFoundError

            frame = cv2.imread(os.path.join(opt.framefolder, "{}.jpg".format(i))) if opt.show or opt.write_video else None

            for f in features:
                features_and_detections[f["ID"]] = {
                    "features": np.array(f["features"], dtype=np.float32)
                }

            for d in detections:
                features_and_detections[int(d["ID"])]["bbox"] = np.array([d["xmin"], d["ymin"], d["xmax"], d["ymax"]], 
                                                                    dtype=np.int32)
            tracker.update(features_and_detections)
            tracklet_manager.update(features_and_detections)

            if opt.show or opt.write_video:
                frame = cv2.imread(os.path.join(opt.framefolder, "{}.jpg".format(i)))
                for track in tracker.tracks:
                    bbox = track.last_bbox
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                    cv2.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (10,100,10),2)
                
                for detkey in features_and_detections:
                    bbox = features_and_detections[detkey]["bbox"]
                    # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(130,130,130), 2)
                    cv2.putText(frame, str(detkey),(int(bbox[0])+30, int(bbox[1])),0, 5e-3 * 200, (160,10,10),2)
                
                # for tracklet_key in tracklet_manager.tracklets:
                #     tracklet = tracklet_manager.tracklets[tracklet_key]
                #     cv2.putText(frame, str(tracklet.tracklet_id),(int(bbox[0])+60, int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

            if opt.show:
                cv2.imshow('', frame)

            if opt.write_video:
                # save a frame
                out.write(frame)
                frame_index = frame_index + 1

            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            if i % 500 == 0:
                print("Frame ", i)
                print("FPS: %f"%(fps))

        except Exception as e:
            # pass
            print("GG", e)
        
        i += 2
        # Press Q to stop!
        if opt.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if opt.write_video:
        out.release()

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--framefolder", type=str, default=".")
    argument_parser.add_argument("--detectionfolder", type=str, default=".")
    argument_parser.add_argument("--featurefolder", type=str, default=".")
    argument_parser.add_argument("--outputfolder", type=str, default=".")
    argument_parser.add_argument("--start", type=int)
    argument_parser.add_argument("--end", type=int)
    argument_parser.add_argument("--show", type=bool, default=False)
    argument_parser.add_argument("--write_video", type=bool, default=False)
    argument_parser.add_argument("--camid", type=int, default=-1)
    opt = argument_parser.parse_args()


    # main(opt)

    
    for min_iou in [0.75, 0.9, 0.8, 0.6, 0.5, 0.95]:
        for max_cosine_distance in [0.5, 0.3, 0.1, 0.05, 0.005]:
            for tracklet_length in [5, 10, 15, 20, 30, 40, 50]:
                if opt.camid > -1 and min_iou==0.75 and max_cosine_distance == 0.5 and tracklet_length == 5:
                    opt.write_video = True
                else:
                    opt.write_video = False

                main(opt, 
                    tracklet_length=tracklet_length,
                    min_iou=min_iou,
                    max_cosine_distance=max_cosine_distance)
