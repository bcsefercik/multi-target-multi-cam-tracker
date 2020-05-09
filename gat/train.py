import os
import pdb
import random

import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from gat import GAT
from utils import EarlyStopping

torch.set_printoptions(threshold=5000)

MARGIN_INTRA = 600
MARGIN_INTER = 6000

def load_data(args):
    generic_filename = 'tl75_iou0.75_d0.3_features.txt'
    frame_width = 1920
    frame_height = 1080

    features = []
    labels = []
    edges = {}  # k: pid v: feature id list

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

                if tracklet_length > 4:
                    bbox = list(map(float, parts[5].strip('][').split(', ')))
                    bbox[0] /= frame_width
                    bbox[1] /= frame_height
                    bbox[2] /= frame_width
                    bbox[3] /= frame_height

                    appearence = list(map(float, parts[-1].strip('][').split(', ')))

                    feature_vector.extend(cam_vector)
                    feature_vector.extend(bbox)
                    feature_vector.extend(appearence)

                    if pid not in edges:
                        edges[pid] = []

                    edges[pid].append({
                        'start': start,
                        'end': end,
                        'cam': cid,
                        'feature_id': len(features)
                    })

                    features.append(feature_vector)
                    labels.append(pid)

    features_ids = list(range(len(features)))
    random.shuffle(features_ids)

    train_mask = [False] * len(features)
    val_mask = [False] * len(features)
    test_mask = [False] * len(features)

    train_count = int(len(features)*0.8)
    val_count = int(len(features)*0.1)
    test_count = len(features) - train_count - val_count

    for i in range(0, train_count):
        train_mask[features_ids[i]] = True

    for i in range(train_count, train_count+val_count):
        val_mask[features_ids[i]] = True

    for i in range(train_count+val_count, train_count+val_count+test_count-1):
        test_mask[features_ids[i]] = True

    g = nx.DiGraph()

    for i in range(len(features)):
        g.add_node(i)

    for k in edges:
        edges[k] = sorted(edges[k], key = lambda x: x['start'])

        for i1 in range(len(edges[k])):
            for i2 in range(i1+1, len(edges[k])):
                time_diff = edges[k][i2]['start'] - edges[k][i1]['end']
                if edges[k][i1]['cam'] != edges[k][i2]['cam']:
                    if time_diff < MARGIN_INTER:
                        g.add_edge(edges[k][i1]['feature_id'], edges[k][i2]['feature_id'])
                        g.add_edge(edges[k][i2]['feature_id'], edges[k][i1]['feature_id'])
                else:
                    if time_diff < MARGIN_INTRA:
                        g.add_edge(edges[k][i1]['feature_id'], edges[k][i2]['feature_id'])
                        g.add_edge(edges[k][i2]['feature_id'], edges[k][i1]['feature_id'])

    return g, features, labels, max(labels)+1, train_mask, val_mask, test_mask





def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        return accuracy(logits, labels)


def main(args):

    with open("output.txt", "a") as fp:
        fp.write(str(args))
        fp.write('\n')
    # load and preprocess dataset
    g, features, labels, num_labels, train_mask, val_mask, test_mask = load_data(args)

    # pdb.set_trace()

    # return
    torch.cuda.empty_cache()
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(train_mask)
        val_mask = torch.BoolTensor(val_mask)
        test_mask = torch.BoolTensor(test_mask)
    else:
        train_mask = torch.ByteTensor(train_mask)
        val_mask = torch.ByteTensor(val_mask)
        test_mask = torch.ByteTensor(test_mask)
    num_feats = features.shape[1]
    n_classes = num_labels
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d 
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    print(model)
    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # loss_fcn = torch.nn.MultiMarginLoss()

    # pdb.set_trace()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)

        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = accuracy(logits[train_mask], labels[train_mask])

        if args.fastmode:
            val_acc = accuracy(logits[val_mask], labels[val_mask])
        else:
            val_acc = evaluate(model, features, labels, val_mask)
            if args.early_stop:
                if stopper.step(val_acc, model):   
                    break

        if epoch%15 == 0:
            torch.save(model, "model.pt")

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))

        with open("output.txt", "a") as fp:
            fp.write("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} |"
              " ValAcc {:.4f} | ETputs(KTEPS) {:.2f}\n".
              format(epoch, np.mean(dur), loss.item(), train_acc,
                     val_acc, n_edges / np.mean(dur) / 1000))

    print()
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))

    

    acc = evaluate(model, features, labels, test_mask)
    print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')

    parser.add_argument("--dataset", type=str, default='.')
    parser.add_argument("--cams", type=int, nargs='+', default=[1, 8])

    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()
    print(args)

    main(args)