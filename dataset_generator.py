import os
import pdb
import random
import pickle
from collections import namedtuple
import argparse

import torch

import networkx as nx
import dgl

from dgl import DGLGraph
torch.set_printoptions(threshold=5000)


GRAPH_SIZE = 12
MARGIN_INTRA = 600
MARGIN_INTER = 6000


def main(args):
    pass


def load_data(args):
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

    # pdb.set_trace()

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

    args = parser.parse_args()

    load_data(args)
