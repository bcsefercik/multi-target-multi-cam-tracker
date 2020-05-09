import argparse
import os


cam_limits = {
    1: (44158, 221998),
    2: (46094, 223934),
    3: (26078, 200297),
    4: (18913, 196114),
    5: (50742, 227540),
    6: (27476, 205139),
    7: (30733, 208573),
    8: (2935, 180775)
}

def main(opt)
    for k in cam_limits:
        cam_path = os.path.join(opt.campath, 'camera{}'.format(k))

        for i in range(1, cam_limits[k][0]):
            try:
                os.remove(os.remove(os.path.join(cam_path, '{}.jpg'.format(i))))
                print('Removed:', k, i)
            except FileNotFoundError:

        for i in range(cam_limits[k][1] + 1, 400000):
            try:
                os.remove(os.remove(os.path.join(cam_path, '{}.jpg'.format(i))))
                print('Removed:', k, i)
            except FileNotFoundError:
                print('Not found:', k, i)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--campath", type=str, default=".")
    opt = argument_parser.parse_args()

    main(opt)
