from __future__ import print_function, absolute_import
import argparse
import os
import os.path as osp
import time
from collections import OrderedDict, defaultdict
import json
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.sampler import RandomIdentitySampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint
from reid.evaluation_metrics import cmc, mean_ap
from reid.feature_extraction import extract_cnn_feature
from reid.utils.meters import AverageMeter


def extract_features(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = defaultdict(list)
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs = extract_cnn_feature(model, imgs)
        for fname, output, pid in zip(fnames, outputs, pids):
            # frame_parts = list(map(int, fname.replace(".jpg", "").split("_")))
            frame_parts = fname.replace(".jpg", "").split("_")
            features[frame_parts[0]].append(
                    {   
                        "ID": frame_parts[2],
                        "xmin": None if len(frame_parts)<7 else int(frame_parts[3]),
                        "ymin": None if len(frame_parts)<7 else int(frame_parts[4]),
                        "xmax": None if len(frame_parts)<7 else int(frame_parts[5]),
                        "ymax": None if len(frame_parts)<7 else int(frame_parts[6]),
                        "confidence": None if len(frame_parts)<8 else float(frame_parts[7]),
                        "features": list(output.numpy().tolist())
                    }
                )
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))

    return features, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="feature extractor")
    parser.add_argument('-d', '--dataset', type=str, default='.')
    parser.add_argument('--imagepath', type=str, default='.')
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int, default=128,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--resume', type=str, default='', metavar='PATH')

    parser.add_argument("--outputpath", type=str, default=".")
    opt = parser.parse_args()

    torch.cuda.empty_cache()

    height = opt.height
    width = opt.width

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    working_dir = osp.dirname(osp.abspath(__file__))
    images_dir = osp.join(working_dir, opt.imagepath)
    root = images_dir # osp.join(images_dir, name)

    test_set = []
    for image in os.listdir(images_dir):
        if ".jpg" in image:
            image_parts = list(map(int, image.replace('.jpg', '').replace('.', '').split("_")))
            test_set.append((image, image_parts[2], image_parts[1]))
    
    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])
    
    test_loader = DataLoader(
        Preprocessor(test_set,
                     root=images_dir, transform=test_transformer),
        batch_size=opt.batch_size, num_workers=opt.workers,
        shuffle=False, pin_memory=True)
    features = None
    with torch.no_grad():
        model = models.create(opt.arch, num_features=1024,
                              dropout=opt.dropout, num_classes=opt.features)
        checkpoint = load_checkpoint(opt.resume)
        model.load_state_dict(checkpoint['state_dict'])
        model = nn.DataParallel(model).cuda()

        features, _ = extract_features(model, test_loader)


    for fid in features:
        with open(os.path.join(opt.outputpath, "{}.json".format(fid)), "w") as f:
            json.dump(features[fid], f, ensure_ascii=False, indent=2)
            print("Frame", fid, "saved.")