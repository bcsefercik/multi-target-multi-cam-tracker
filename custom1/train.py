import os
import pdb
import random
import pickle
from collections import namedtuple
import argparse

import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dgl import DGLGraph
from model import BCS
from utils import EarlyStopping

from dgl.data import SSTBatch
import dgl
torch.set_printoptions(threshold=5000)

MARGIN_INTRA = 600
MARGIN_INTER = 6000
GRAPH_SIZE = 12

BATCH_SIZE = 300

BCSBatch = namedtuple('BCSBatch', ['graph'])

# Used for creating datafile from tacklet records, for once.
def load_data(args):
    frame_width = 1920
    frame_height = 1080
    dataset = None

    with open(args.datafile, 'rb') as f:
        dataset = pickle.load(f)

    features = dataset['features']
    data = dataset['data']

    x = []
    graphs = []
    graph_data = []
    tracklet_features = []
    labels = []
    graph_nodes = []

    for k in data:
        if len(data[k]) > GRAPH_SIZE:
            data[k] = sorted(data[k], key = lambda x: x['start'])

            for i1 in range(len(data[k]) - GRAPH_SIZE):
                g = nx.DiGraph()
                g_nodes = []
                graph_features = [None] * GRAPH_SIZE
                graph_features[0] = features[data[k][i1]['feature_id']]

                for i in range(GRAPH_SIZE):
                    g.add_node(i)

                # for i in range(GRAPH_SIZE):
                #     for j in range(GRAPH_SIZE):
                #         g.add_edge(i, j)
                #         g.add_edge(j, i)

                for i2 in range(i1+1, i1+GRAPH_SIZE):
                    ii = i2 - i1
                    graph_features[ii] = features[data[k][i2]['feature_id']]
                    time_diff = data[k][i2]['start'] - data[k][i1]['end']
                    if data[k][i1]['cam'] != data[k][i2]['cam']:
                        if time_diff < MARGIN_INTER:
                            g.add_edge(0, ii)
                            g.add_edge(ii, 0)
                    else:
                        if time_diff < MARGIN_INTRA:
                            g.add_edge(0, ii)
                            g.add_edge(ii, 0)

                    for i3 in range(i2+1, i1+GRAPH_SIZE):
                        iii = i3 - i1
                        time_diff = data[k][i3]['start'] - data[k][i2]['end']
                        if data[k][i2]['cam'] != data[k][i3]['cam']:
                            if time_diff < MARGIN_INTER:
                                g.add_edge(iii, ii)
                                g.add_edge(ii, iii)
                        else:
                            if time_diff < MARGIN_INTRA:
                                g.add_edge(iii, ii)
                                g.add_edge(ii, iii)

                        g_nodes.append(data[k][i2]['cam'])
                        g_nodes.append(data[k][i3]['cam'])

                g.remove_edges_from(nx.selfloop_edges(g))
                g = DGLGraph(g)
                g.add_edges(g.nodes(), g.nodes())

                positive_features = features[random.choice(data[k][i1+GRAPH_SIZE:])['feature_id']]

                negative_features = None
                while True:
                    r = random.choice(list(data.keys()))

                    if r == k:
                        continue

                    negative_features = features[random.choice(data[r])['feature_id']]

                    break

                x.append((g, torch.FloatTensor(graph_features), torch.FloatTensor(positive_features)))
                graphs.append(g)
                graph_data.append(graph_features)
                tracklet_features.append(positive_features)
                labels.append(1)

                x.append((g, torch.FloatTensor(graph_features), torch.FloatTensor(negative_features)))
                graphs.append(g)
                graph_data.append(graph_features)
                tracklet_features.append(negative_features)
                labels.append(-1)

                graph_nodes.append(g_nodes)

    # return graph_nodes
    # pdb.set_trace()


    set_ids = list(range(len(labels)))
    random.shuffle(set_ids)

    training_set_limit = int(len(set_ids)*0.9)

    training_output_data = {
        'x': x[:training_set_limit],
        'labels': torch.LongTensor(labels[:training_set_limit]),
        'set_ids': set_ids[:training_set_limit],
        'graphs': graphs[:training_set_limit],
        'graph_data': graph_data[:training_set_limit],
        'tracklet_features': tracklet_features[:training_set_limit]
    }

    val_output_data = {
        'x': x[training_set_limit:],
        'labels': torch.LongTensor(labels[training_set_limit:]),
        'set_ids': set_ids[training_set_limit:],
        'graphs': graphs[training_set_limit:],
        'graph_data': graph_data[training_set_limit:],
        'tracklet_features': tracklet_features[training_set_limit:]
    }

    with open(args.trainingdataset, 'wb') as f:
        pickle.dump(training_output_data, f, pickle.HIGHEST_PROTOCOL)

    with open(args.valdataset, 'wb') as f:
        pickle.dump(val_output_data, f, pickle.HIGHEST_PROTOCOL)

    return training_output_data, val_output_data


def accuracy(logits, labels):
    # pdb.set_trace()
    # pred = torch.BoolTensor(logits>0).cuda()
    # gold = torch.BoolTensor(labels>0).cuda()
    pred = logits>0
    gold = labels>0
    correct = torch.sum(pred == gold)
    return correct.item() * 1.0 / len(labels)
    # return 1.0

def evaluate(model, labels, batch, node_features, tracklet_features):
    model.eval()
    with torch.no_grad():
        logits = model(batch, node_features, tracklet_features)
        return accuracy(logits, labels)

def batcher(dev):
    def batcher_dev(batch):
        # pdb.set_trace()
        batch_trees = dgl.batch(batch)
        return BCSBatch(graph=batch_trees)
    return batcher_dev

def main(args):
    best_val_acc = 0
    dataset_name = args.trainingdataset.split("_training.pickle")[0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # graph_nodes = load_data(args)
    # graph_sets = [None]*len(graph_nodes)
    # for i, v in enumerate(graph_nodes):
    #     graph_sets[i] = set(v)
    
    # set_lengths = list(map(len, graph_sets))

    # ones = list(filter(lambda x: x == 1, set_lengths))
    # twos = list(filter(lambda x: x == 2, set_lengths))
    # threes = list(filter(lambda x: x == 3, set_lengths))

    # pdb.set_trace()
    # return

    with open("{}_output.txt".format(dataset_name), "w") as fp:
        fp.write(str(args))
        fp.write('\n')

    if args.datafile:
        load_data(args)
        return

    dataset_training = None
    with open(args.trainingdataset, 'rb') as f:
        dataset_training = pickle.load(f)
    
    dataset_val = None
    with open(args.valdataset, 'rb') as f:
        dataset_val = pickle.load(f)

    ####### DATASET KEYS
    # 'labels': 1/-1,
    # 'set_ids': shuffled ids,
    # 'graphs': graphs
    # 'graph_data': graph node features
    # 'tracklet_features': tracklet_features


    # training_set_limit = int(len(dataset['set_ids'])*0.85)

    training_graph_data = []
    training_tracklet_data = []
    training_graphs = []
    training_labels = []

    test_graph_data = []
    test_tracklet_data = []
    test_graphs = []
    test_labels = []

    training_set_ids = dataset_training['set_ids']
    random.shuffle(training_set_ids)
    # test_set_ids = list(range(training_set_limit, len(dataset['set_ids'])))
    test_set_ids = dataset_val['set_ids']

    # print(len(dataset_training['graphs']))
    # print(len(dataset_val['graphs']))

    for i in range(len(dataset_training['labels'])):
        training_graphs.append(dataset_training['graphs'][i])
        training_graph_data.append(dataset_training['graph_data'][i])
        training_tracklet_data.append(dataset_training['tracklet_features'][i])
        training_labels.append(dataset_training['labels'][i])

    for i in range(len(dataset_val['labels'])):
        test_graph_data.append(dataset_val['graph_data'][i])
        test_tracklet_data.append(dataset_val['tracklet_features'][i])
        test_graphs.append(dataset_val['graphs'][i])
        test_labels.append(dataset_val['labels'][i])

    print(len(training_labels), len(training_labels)/args.batchsize)

    training_labels = torch.FloatTensor(training_labels)
    training_graph_data = torch.FloatTensor(training_graph_data)
    training_tracklet_data = torch.FloatTensor(training_tracklet_data)
    test_labels = torch.FloatTensor(test_labels)
    test_graph_data = torch.FloatTensor(test_graph_data)
    test_tracklet_data = torch.FloatTensor(test_tracklet_data)
    

    train_loader = DataLoader(
        dataset=training_graphs,
        batch_size=args.batchsize,
        collate_fn=batcher(device),
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        dataset=test_graphs,
        batch_size=args.batchsize,
        collate_fn=batcher(device),
        shuffle=False,
        num_workers=0
    )



    if args.gpu < 0:
        cuda = False
    else:
        torch.cuda.empty_cache()
        cuda = True
        torch.cuda.set_device(args.gpu)
        # training_graphs = training_graphs.cuda()
        training_graph_data = training_graph_data.cuda()
        training_tracklet_data = training_tracklet_data.cuda()
        training_labels = training_labels.cuda()

        # test_graphs = test_graphs.cuda()
        test_graph_data = test_graph_data.cuda()
        test_tracklet_data = test_tracklet_data.cuda()
        test_labels = test_labels.cuda()
    
    # create model

    model = None
    if os.path.exists("{}_model.pt".format(dataset_name)):
        model = torch.load("{}_model.pt".format(dataset_name))
    else:
        model = BCS()
    # model = torch.load("model.pt")

    if args.early_stop:
        stopper = EarlyStopping(patience=100)
    if cuda:
        model = model.cuda()

    loss_func = torch.nn.SoftMarginLoss()
    # loss_func = torch.nn.MultiMarginLoss()
    # loss_func = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dur = []
    loss_values = []
    training_accuracy = []
    val_accuracy = []
    for epoch in range(args.epochs):
        training_accuracy.append([])
        val_accuracy.append([])
        loss_values.append([])
        total_count = 0
        # print(len(training_graphs), len(training_graph_data), len(training_tracklet_data))

        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward

        for step, batch in enumerate(train_loader):
            if len(training_graphs) < (step+1)*args.batchsize:
                continue
            node_features = training_graph_data[step*args.batchsize:min(len(training_graphs), (step+1)*args.batchsize), :]
            tracklet_features = training_tracklet_data[step*args.batchsize:min(len(training_graphs), (step+1)*args.batchsize), :]
            logits = model(batch, node_features, tracklet_features)
            loss = loss_func(logits, training_labels[step*args.batchsize:min(len(training_graphs), (step+1)*args.batchsize)])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_accuracy[epoch].append(
                accuracy(logits, training_labels[step*args.batchsize:min(len(training_graphs), (step+1)*args.batchsize)]))
            loss_values[epoch].append(loss.item())

            log_str = "Epoch {:05d} | Batch {:05d} | Loss {:.4f} | Acc {:.4f}\n".format(
                        epoch, 
                        step, 
                        loss_values[epoch][-1], 
                        training_accuracy[epoch][-1], 
                    )

            print(log_str.strip())
            with open("{}_output.txt".format(dataset_name), "a") as fp:
                fp.write(log_str)

        for step, batch in enumerate(test_loader):
            if len(test_graphs) < (step+1)*args.batchsize:
                continue
            node_features = test_graph_data[step*args.batchsize:min(len(test_graphs), (step+1)*args.batchsize), :]
            tracklet_features = test_tracklet_data[step*args.batchsize:min(len(test_graphs), (step+1)*args.batchsize), :]
            
            val_accuracy[epoch].append(evaluate(
                model, 
                test_labels[step*args.batchsize:min(len(test_graphs), (step+1)*args.batchsize)],
                batch,
                node_features,
                tracklet_features
            ))


        if epoch >= 3:
            dur.append(time.time() - t0)


        # if args.fastmode:
        #     val_acc = accuracy(logits[val_mask], labels[val_mask])
        # else:
        #     val_acc = evaluate(model, features, labels, val_mask)
        #     if args.early_stop:
        #         if stopper.step(val_acc, model):   
        #             break

        train_acc = np.mean(training_accuracy[epoch])
        val_acc = np.mean(val_accuracy[epoch])

        print(train_acc, val_acc)

        if epoch%3 == 0 and val_acc > best_val_acc:
            torch.save(model, "{}_model.pt".format(dataset_name))

        log_str = "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | TrainAcc {:.4f} | TestAcc {:.4f}\n".format(
            epoch, 
            np.mean(dur), 
            float(np.mean(loss_values[epoch])), 
            float(train_acc), 
            float(val_acc)
        )

        print(log_str)

        with open("{}_output.txt".format(dataset_name), "a") as fp:
            fp.write(log_str)

    print()
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')

    parser.add_argument("--dataset", type=str, default='dataset.pickle')
    parser.add_argument("--trainingdataset", type=str, default='tl120_iou0.85_d0.3_processed_training.pickle')
    parser.add_argument("--valdataset", type=str, default='tl120_iou0.85_d0.3_processed_val.pickle')
    parser.add_argument("--output", type=str, default='dataset.pickle')
    parser.add_argument("--datafile", type=str, default=None)
    parser.add_argument("--cams", type=int, nargs='+', default=[1, 8])
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument("--batchsize", type=int, default=512)

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