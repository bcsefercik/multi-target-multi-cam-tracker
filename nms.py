import argparse
import os

from collections import defaultdict
import json

import numpy as np

def non_max_suppression(detections, max_bbox_overlap, confidence_threshold=0.01):
    if len(detections) == 0:
        return []

    boxes = np.array([[d["xmin"], d["ymin"], d["xmax"], d["ymax"]] for d in detections])
    scores = np.array([d["confidence"] for d in detections])

    boxes = boxes.astype(np.float)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return [detections[i] for i in pick if detections[i]["confidence"]>confidence_threshold]


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--inputpath", type=str, default=".")
    argument_parser.add_argument("--outputpath", type=str, default=".")
    argument_parser.add_argument("--confidence", type=float, default=0.1)
    argument_parser.add_argument("--nms", type=float, default=0.5)
    opt = argument_parser.parse_args()

    i = 1
    while True:
        try:
            file_path = os.path.join(opt.inputpath, "{}.json".format(i))

            if not os.path.isfile(file_path):
                break

            data = None

            with open(file_path, "r") as f:
                data = json.load(f)

            print(data)

            output_data = non_max_suppression(data, max_bbox_overlap=opt.nms, confidence_threshold=opt.confidence)

            with open(os.path.join(opt.outputpath, "{}.json".format(i)), "w") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
                print("Frame", i, "saved.")
        except Exception as e:
            print("gggg")

        i += 1
