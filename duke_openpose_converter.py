import argparse
import os

from collections import defaultdict
import json

# Cam ID:  1  Min:  44158 Max: 221998 Diff: 177840    (+)Frames: 164503   Max Gap:   2358
# Cam ID:  2  Min:  46094 Max: 223934 Diff: 177840    (+)Frames: 169974   Max Gap:   1567
# Cam ID:  3  Min:  26078 Max: 200297 Diff: 174219    (+)Frames: 113122   Max Gap:   3229
# Cam ID:  4  Min:  18913 Max: 196114 Diff: 177201    (+)Frames: 132381   Max Gap:   3994
# Cam ID:  5  Min:  50742 Max: 227540 Diff: 176798    (+)Frames: 135762   Max Gap:   4276
# Cam ID:  6  Min:  27476 Max: 205139 Diff: 177663    (+)Frames: 168728   Max Gap:   1998
# Cam ID:  7  Min:  30733 Max: 208573 Diff: 177840    (+)Frames: 130566   Max Gap:   3726
# Cam ID:  8  Min:   2935 Max: 180775 Diff: 177840    (+)Frames: 159112   Max Gap:   2104

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--inputpath", type=str, default=".")
    argument_parser.add_argument("--outputpath", type=str, default=".")
    argument_parser.add_argument("--width", type=int, default=1920)
    argument_parser.add_argument("--height", type=int, default=1080)
    argument_parser.add_argument("--confidence", type=float, default=0.1)
    opt = argument_parser.parse_args()

    i = 1
    while True:
        try:
            file_path = os.path.join(opt.inputpath, "{}_keypoints.json".format(i))

            if not os.path.isfile(file_path):
                break

            openpose_data = None
            output_data = []

            with open(file_path, "r") as f:
                openpose_data = json.load(f)


            for pid, p in enumerate(openpose_data["people"]):
                xmin = 1.0
                ymin = 1.0
                xmax = 0.0
                ymax = 0.0
                conf = 1.0

                for pi in range(0, len(p["pose_keypoints"]), 3):
                    if p["pose_keypoints"][pi+2] > opt.confidence:
                        xmin = min(p["pose_keypoints"][pi], xmin)
                        ymin = min(p["pose_keypoints"][pi+1], ymin)
                        xmax = max(p["pose_keypoints"][pi], xmax)
                        ymax = max(p["pose_keypoints"][pi+1], ymax)
                        conf = min(p["pose_keypoints"][pi+2], conf)

                if xmin < xmax and ymin < ymax:
                    output_data.append(
                        {
                            "xmin": int(xmin*opt.width),
                            "ymin": int(ymin*opt.height),
                            "xmax": int(xmax*opt.width),
                            "ymax": int(ymax*opt.height),
                            "confidence": conf,
                            "ID": pid
                        }
                    )

            with open(os.path.join(opt.outputpath, "{}.json".format(i)), "w") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            print("Frame", i, "saved.")
        except Exception as e:
            print(e)

        i += 1
