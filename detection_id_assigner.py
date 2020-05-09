import argparse
import os
import json

import numpy as np

def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""
    xmin = np.maximum(b1[:,0], b2[:,0])
    ymin = np.maximum(b1[:,1], b2[:,1])
    xmax = np.minimum(b1[:,0] + b1[:,2], b2[:,0] + b2[:,2])
    ymax = np.minimum(b1[:,1] + b1[:,3], b2[:,1] + b2[:,3])
    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)
    return width * height

def area2d(b):
    return np.multiply(b[:,2], b[:,3])

def iou2d(b1, b2):
    """Compute the IoU between a set of boxes b1 and 1 box b2"""
    if b1.ndim == 1: b1 = b1[None, :]
    if b2.ndim == 1: b2 = b2[None, :]
    assert b2.shape[0] == 1
    ov = overlap2d(b1, b2)
    return ov / (area2d(b1) + area2d(b2) - ov)

def get_labels(det_xywh, gt_xywh, tid, iou1=0.3, iou2=0.31):
    labels = np.zeros(det_xywh.shape[0])
    label_max_iou = iou2 * np.ones(det_xywh.shape[0])
    for i, xywh in enumerate(det_xywh):
        iou = iou2d(gt_xywh, xywh)
        if not iou.size: continue
        gt_iou1 = iou > iou1
        if gt_iou1.sum() == 0:
            labels[i] = -1
        max_idx = np.argmax(iou)
        max_iou = iou[max_idx]
        if max_iou >= label_max_iou[i]:
            labels[i] = tid[max_idx]
            label_max_iou[i] = max_iou
    return labels.astype(np.int32), label_max_iou

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--gtfolder", type=str, default=".")
    argument_parser.add_argument("--detfolder", type=str, default=".")
    argument_parser.add_argument("--outfolder", type=str, default=".")
    opt = argument_parser.parse_args()

    i = 1
    while True:
        try:
            gt_file = os.path.join(opt.gtfolder, "{}.json".format(i))
            det_file = os.path.join(opt.detfolder, "{}.json".format(i))

            if not os.path.isfile(gt_file):
                if i < 400000:
                    with open(os.path.join(opt.outfolder, "{}.json".format(i)), "w") as f:
                        json.dump([], f)
                    i += 1
                    continue
                else:
                    break

            gt_data = None
            det_data = None

            with open(gt_file, "r") as fgt, open(det_file, "r") as fdet:
                gt_data = json.load(fgt)
                det_data = json.load(fdet)

            gt_xywh = np.array(list(map(lambda a: np.array([a["xmin"], a["ymin"], a["xmax"]-a["xmin"], a["ymax"]-a["ymin"]]), gt_data)))
            det_xywh = np.array(list(map(lambda a: np.array([a["xmin"], a["ymin"], a["xmax"]-a["xmin"], a["ymax"]-a["ymin"]]), det_data)))
            tid = np.array(list(map(lambda a: a["ID"], gt_data)))

            labels, ious = get_labels(det_xywh, gt_xywh, tid)

            for li, label in enumerate(labels):
                det_data[li]["ID"] = int(label) if label > 0 else -1

            with open(os.path.join(opt.outfolder, "{}.json".format(i)), "w") as f:
                json.dump(det_data, f, ensure_ascii=False, indent=2)

            print("Frame", i, "saved.")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(e)

        i += 1
