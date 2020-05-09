import os
import pdb
import random

import argparse
import numpy as np
import pickle

NODE_COUNT = 4

def load_data(args):
    generic_filename = args.generic
    frame_width = 1920
    frame_height = 1080

    features = []
    labels = []
    data = {}  # k: pid v: feature id list

    for cid in range(args.cams[0], args.cams[1]+1):
        file_path = os.path.join(args.dataset, 'camera{}'.format(cid), generic_filename)
        cam_vector = [0] * (args.cams[1]-args.cams[0]+1)
        cam_vector[cid-1] = 1

        with open(file_path) as fp:
            while True:
                line = fp.readline()

                if not line:
                    break

                feature_vector = []

                line = line.strip()

                parts = line.split('\t')

                pid = int(parts[0])
                tracklet_length = int(parts[2])
                start = int(parts[3])
                end = int(parts[4])

                if tracklet_length > 2:
                    bbox = list(map(float, parts[5].strip('][').split(', ')))
                    bbox[0] /= frame_width
                    bbox[1] /= frame_height
                    bbox[2] /= frame_width
                    bbox[3] /= frame_height

                    appearence = list(map(float, parts[-1].strip('][').split(', ')))

                    # feature_vector.extend(cam_vector)
                    # feature_vector.extend(bbox)
                    feature_vector.extend(appearence)

                    if pid not in data:
                        data[pid] = []

                    data[pid].append({
                        'start': start,
                        'end': end,
                        'cam': cid,
                        'feature_id': len(features)
                    })

                    features.append(feature_vector)
                    labels.append(pid)
    set_ids = list(range(len(features)))
    random.shuffle(set_ids)

    return data, features, labels, set_ids

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DataLoader')

    parser.add_argument("--dataset", type=str, default='.')
    parser.add_argument("--generic", type=str, default='tl10_iou0.75_d0.3_features.txt')
    parser.add_argument("--cams", type=int, nargs='+', default=[1, 8])
    parser.add_argument("--output", type=str, default='dataset.pickle')
    args = parser.parse_args()
    
    data, features, labels, set_ids = load_data(args)

    dataset = {
        'data': data,
        'features': features,
        'labels': labels,
        'set_ids': set_ids
    }

    with open(args.output, 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

    # datas = None
    # with open(args.output, 'rb') as f:
    #     datas = pickle.load(f)
    # print(datas)
