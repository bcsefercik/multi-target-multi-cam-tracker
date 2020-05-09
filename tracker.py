import numpy as np

from scipy.optimize import linear_sum_assignment


from track import Track
from track import TrackStatus

class Tracker:
    def __init__(self, min_iou=0.7, max_cosine_distance=0.3):
        self.min_iou = min_iou
        self.max_cosine_distance = max_cosine_distance
        self.tracks = []
        self.next_track_id = 1

    def init_track(self, features, bbox):
        new_track = Track(
                self.next_track_id, 
                features, 
                bbox
                )
        new_track.is_dirty = True
        self.tracks.append(new_track)

        self.next_track_id += 1

    def clean_tracks(self):
        for i, _ in enumerate(self.tracks):
            self.tracks[i].is_dirty = False

    def update(self, features_and_detections):
        if not self.tracks:
            for dkey in features_and_detections:
                self.init_track(
                    features_and_detections[dkey]["features"], 
                    features_and_detections[dkey]["bbox"]
                    )
        else:
            cost_matrix = []
            index_matches = []
            for dkey in features_and_detections:
                index_matches.append(dkey)
                cosine_distances = [100.0] * len(self.tracks)

                for i, track in enumerate(self.tracks):
                    cosine_distances[i] = self.compute_cosine_dist(features_and_detections[dkey]["features"], track.features)
                    if cosine_distances[i] > self.max_cosine_distance:
                        cosine_distances[i] += 100

                cost_matrix.append(cosine_distances)

            if len(cost_matrix) > 0:
                cost_matrix = np.array(cost_matrix, dtype=np.float32)

                indices = linear_sum_assignment(cost_matrix)
                # rows: detections
                # columns: tracks
                for ij in range(len(indices[0])):
                    detection_index = indices[0][ij]
                    track_index = indices[1][ij]
                    # print(self.compute_iou(   features_and_detections[index_matches[detection_index]]["bbox"], 
                    #                             self.tracks[track_index].last_bbox))
                    if (not self.tracks[track_index].is_dirty) \
                        and cost_matrix[detection_index][track_index] < self.max_cosine_distance \
                        and self.compute_iou(   features_and_detections[index_matches[detection_index]]["bbox"], 
                                                self.tracks[track_index].last_bbox)>self.min_iou:
                        self.tracks[track_index].update(
                            features_and_detections[index_matches[detection_index]]["features"],
                            features_and_detections[index_matches[detection_index]]["bbox"])

                        index_matches[detection_index] = -1

            for ii, vv in enumerate(index_matches):
                if vv != -1:
                    # print("init")
                    self.init_track(
                        features_and_detections[vv]["features"], 
                        features_and_detections[vv]["bbox"]
                    )

            for ii, vv in enumerate(self.tracks):
                if not vv.is_dirty:
                    self.tracks[ii] = -1

            for _ in range(self.tracks.count(-1)):
                self.tracks.remove(-1)

        # print(self.tracks)
                
        self.clean_tracks()

    def compute_iou(self, boxA, boxB):
        if isinstance(boxA, type(None)) or isinstance(boxB, type(None)):
            return 1.0
            
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        
        return iou

    def area2d(self, b):
        return (b[2]-b[0])*(b[3]-b[1])

    def iou2d(self, b1, b2):
        ov = self.overlap2d(b1, b2)
        return ov / (self.area2d(b1) + self.area2d(b2) - ov)

    def compute_cosine_dist(self, emb1, emb2):
        sim = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
        return 1.0-float(sim)