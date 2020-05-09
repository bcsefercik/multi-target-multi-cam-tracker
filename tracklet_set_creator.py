import os
import argparse

import numpy as np


class Tracklet:
    def __init__(self, ID, frame_id, bbox, features, max_length):
        self.ID = ID
        self.max_length = max_length
        self.x_values = [bbox[0]]
        self.y_values = [bbox[1]]
        self.w_values = [bbox[2] - bbox[0]]
        self.h_values = [bbox[3] - bbox[1]]
        self.features = features
        self.age = 1
        self.entrance = frame_id
        self.exit = frame_id
        self.frame_ids = [frame_id]

    def update(self, frame_id, bbox, features):
        self.exit = frame_id
        self.age += 1

        # self.features = np.maximum(self.features, features)
        self.features = ((self.age - 1)*self.features + features) / self.age
        
        self.x_values.append(new_bbox[0])
        self.y_values.append(new_bbox[1])
        self.w_values.append(new_bbox[2] - new_bbox[0])
        self.h_values.append(new_bbox[3] - new_bbox[1])

    
    def is_done(self):
        return self.max_length == self.age

    def get_purity(self):
        return max(self.predicted_ids.values())/self.max_length if len(self.predicted_ids) > 1 and (-1 not in self.predicted_ids) else 1.0

    def get_id(self):
        return max(self.predicted_ids, key=self.predicted_ids.get)

    def get_bbox(self):
        result = [None] * 4
        result[0] = int(statistics.median(self.x_values))
        result[1] = int(statistics.median(self.y_values))
        result[2] = int(statistics.median(self.w_values))
        result[3] = int(statistics.median(self.h_values))

        return result


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--inputpath", type=str, default=".")
    argument_parser.add_argument("--outputpath", type=str, default=".")
    argument_parser.add_argument("--width", type=int, default=1920)
    argument_parser.add_argument("--height", type=int, default=1080)
    argument_parser.add_argument("--confidence", type=float, default=0.1)
    opt = argument_parser.parse_args()

    maind(opt)
