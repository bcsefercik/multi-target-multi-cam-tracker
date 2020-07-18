import os
import time
import json
import argparse
import statistics

import numpy as np

from nms import non_max_suppression


class Tracklet:
    def __init__(self, ID, frame_id, bbox, features, max_length, forget_length):
        self.ID = ID
        self.max_length = max_length
        self.forget_length = forget_length
        self.x_values = [bbox[0]]
        self.y_values = [bbox[1]]
        self.w_values = [bbox[2] - bbox[0]]
        self.h_values = [bbox[3] - bbox[1]]
        self.features = features
        self.length = 1
        self.entrance = frame_id
        self.exit = frame_id
        self.frame_ids = [frame_id]

    def update(self, frame_id, bbox, features):
        self.exit = frame_id
        self.length += 1

        # self.features = np.maximum(self.features, features)
        self.features = ((self.length - 1)*self.features + features) / self.length
        
        self.x_values.append(bbox[0])
        self.y_values.append(bbox[1])
        self.w_values.append(bbox[2] - bbox[0])
        self.h_values.append(bbox[3] - bbox[1])

        self.frame_ids.append(frame_id)
 
    def is_done(self, frame_id):
        return self.length >= self.max_length or (frame_id - self.exit) > self.forget_length

    def get_bbox(self):
        result = [None] * 4
        result[0] = int(statistics.median(self.x_values))
        result[1] = int(statistics.median(self.y_values))
        result[2] = int(statistics.median(self.w_values))
        result[3] = int(statistics.median(self.h_values))

        return result


class TrackletManager():
    def __init__(self, max_tracklet_length, file_path, cam_id=1, forget_threshold=3):   
        self.max_tracklet_length = max_tracklet_length
        self.cam_id = cam_id
        self.forget_threshold = forget_threshold
        self.tracklets = {}

        self.file_path = file_path 
        self.file_path_features = self.file_path.replace('.txt', '')
        self.file_path_features = '{}_features.txt'.format(self.file_path_features)

    def update(self, features_and_detections, frame_id):
        keys_to_pop = set()
        popped_tracklets = []

        for ID in features_and_detections:
            if ID not in self.tracklets:
                self.tracklets[ID] = Tracklet(
                    ID,
                    frame_id, 
                    list(features_and_detections[ID]["bbox"]),
                    features_and_detections[ID]["features"], 
                    self.max_tracklet_length,
                    self.forget_threshold
                )
            else:
                self.tracklets[ID].update(
                    frame_id,
                    list(features_and_detections[ID]["bbox"]),
                    features_and_detections[ID]["features"]
                )

                if self.tracklets[ID].is_done(frame_id):
                    keys_to_pop.add(ID)

        for ID in keys_to_pop:
            features_log_text = '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                self.tracklets[ID].ID,
                1,  # Purity is always 1
                self.tracklets[ID].length,
                self.tracklets[ID].entrance,
                self.tracklets[ID].exit,
                self.tracklets[ID].get_bbox(),
                self.tracklets[ID].features.tolist()
            )
            log_text = f"{ID}\t{1}\t{self.tracklets[ID].length}\n"

            # print(features_log_text)

            with open(self.file_path, "a") as f, open(self.file_path_features, "a") as ff:
                f.write(log_text)
                ff.write(features_log_text)

            popped_tracklets.append(self.tracklets.pop(ID))

        return popped_tracklets

def main(opt, min_iou, max_cosine_distance, tracklet_length):
    tracklet_manager = TrackletManager(
            tracklet_length, 
            os.path.join(
                opt.outputfolder, 
                "tl{}_iou{}_d{}.txt".format(
                    tracklet_length, 
                    min_iou, 
                    max_cosine_distance
                )
            ), 
            cam_id=opt.camid,
            forget_threshold=opt.forget
        )

    i = opt.start
    fps = 0.0
    while i <= opt.end:
        try:
            t1 = time.time()

            features_and_detections = {}
            features = []

            try:
                with open(os.path.join(opt.featurefolder, "{}.json".format(i))) as f:
                    features = json.load(f)
            except FileNotFoundError:
                print("INFO: No features for frame", i)
                # raise FileNotFoundError

            features = non_max_suppression(features, 0.5, confidence_threshold=0.11)

            for f in features:
                if int(f["ID"]) > 0:
                    features_and_detections[int(f["ID"])] = {
                        "features": np.array(f["features"], dtype=np.float32),
                        "bbox": np.array([f["xmin"], f["ymin"], f["xmax"], f["ymax"]], dtype=np.int32)
                    }

            tracklet_manager.update(features_and_detections, i)

            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            if i % 500 == 0:
                print("Frame ", i)
                print("FPS: %f"%(fps))

        except Exception as e:
            # pass
            print("GG", e)
        
        i += 1


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()

    argument_parser.add_argument("--detectionfolder", type=str, default=".")
    argument_parser.add_argument("--featurefolder", type=str, default=".")
    argument_parser.add_argument("--outputfolder", type=str, default=".")
    argument_parser.add_argument("--start", type=int)
    argument_parser.add_argument("--end", type=int)
    argument_parser.add_argument("--camid", type=int, default=1)
    argument_parser.add_argument("--forget", type=int, default=5)
    argument_parser.add_argument('--trackletlengths', nargs='+', type=int)
    argument_parser.add_argument('--ious', nargs='+', type=float)
    argument_parser.add_argument('--distances', nargs='+', type=float)

    opt = argument_parser.parse_args()

    for min_iou in opt.ious:
        for max_cosine_distance in opt.distances:
            for tracklet_length in opt.trackletlengths:
                if opt.camid > -1 and min_iou==0.75 and max_cosine_distance == 0.5 and tracklet_length == 5:
                    opt.write_video = True
                else:
                    opt.write_video = False

                main(opt, 
                    tracklet_length=tracklet_length,
                    min_iou=min_iou,
                    max_cosine_distance=max_cosine_distance)