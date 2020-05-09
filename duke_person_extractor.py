import scipy.io
import cv2
import argparse
import os
import json

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--inputpath", type=str, default=".")
    argument_parser.add_argument("--outputpath", type=str, default=".")
    argument_parser.add_argument("--camid", type=int)
    argument_parser.add_argument("--framefolder", type=str, default=".")
    argument_parser.add_argument("--start", type=int, default=1)

    opt = argument_parser.parse_args()

    for i in range(1, 41):
        folder_path = os.path.join(opt.outputpath, str(i))
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    i = opt.start
    while True:
        sub_folder = str(min(round(i/10000)+1, 40))
        file_path = os.path.join(opt.inputpath, "{}.json".format(i))

        if not os.path.isfile(file_path):
            break

        data = None

        with open(file_path, "r") as f:
            data = json.load(f)

        for pi, p in enumerate(data):
            output_filename = str(os.path.join(opt.outputpath, sub_folder, "{}_{}_{}_{}_{}_{}_{}_{}.jpg".format(
                i, 
                opt.camid, 
                p["ID"] if int(p["ID"] > 0) else -pi, 
                p["xmin"], 
                p["ymin"], 
                p["xmax"], 
                p["ymax"], 
                p.get("confidence", 1.0))))
            frame_path = str(os.path.join(opt.framefolder, "{}.jpg".format(i)))

            if os.path.isfile(frame_path):
                img = cv2.imread(frame_path)
                cv2.imwrite(output_filename, img[p["ymin"]:p["ymax"], p["xmin"]:p["xmax"]])
                
                if i % 1000 == 0:
                    print(output_filename, "written.")



        i += 1