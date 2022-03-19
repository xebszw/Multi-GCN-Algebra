from os.path import join as pjoin
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
import torch.utils
import torch
import math
import time
import copy
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('agg') # 防止linux服务器报错
print('using torch', torch.__version__)

# Experiment parameters
# argparse 模块是 Python 内置的一个用于命令项选项与参数解析的模块
parser = argparse.ArgumentParser(description='Graph Convolutional Networks')
parser.add_argument('-D', '--dataset', type=str, default='BEHAVIOR')
parser.add_argument('-M', '--model', type=str, default='gcn',
                    choices=['gcn', 'unet', 'mgcn'])
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--lr_decay_steps', type=str, # 衰减周期
                    default='25,35', help='learning rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay') # 权重衰减

# Dropout: 防止过拟合
# Dropout说的简单一点就是：我们在前向传播的时候，让某个神经元的激活值以一定的概率p停止工作，
# 这样可以使模型泛化性更强，因为它不会太依赖某些局部的特征
parser.add_argument('-d', '--dropout', type=float,
                    default=0.1, help='dropout rate')

# 我们指定了多少的filters那么就是把一张图片变成了多少个通道的图
parser.add_argument('-f', '--filters', type=str,
                    default='64,64,64', help='number of filters in each layer')
parser.add_argument('-K', '--filter_scale', type=int, default=1,
                    help='filter scale (receptive field size), must be > 0; 1 for GCN, >1 for ChebNet')

# 卷积层后的全连接层中隐藏单元的数量
parser.add_argument('--n_hidden', type=int, default=0,
                    help='number of hidden units in a fully connected layer after the last conv layer')

# 边预测网络的全连接层中隐藏单元数量
parser.add_argument('--n_hidden_edge', type=int, default=32,
                    help='number of hidden units in a fully connected layer of the edge prediction network')

# 加了该参数则触发action，值变为true，否则默认为false
parser.add_argument('--degree', action='store_true',
                    default=False, help='use one-hot node degree features')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs')
parser.add_argument('-b', '--batch_size', type=int,
                    default=32, help='batch size')

# 批规范化，解决训练数据分布变化问题                    
parser.add_argument('--bn', action='store_true',
                    default=False, help='use BatchNorm layer')
parser.add_argument('--folds', type=int, default=1,
                    help='number of cross-validation folds (1 for COLORS and TRIANGLES and 10 for other datasets)')

# 线程
parser.add_argument('-t', '--threads', type=int, default=0,
                    help='number of threads to load data')

# 日志记录间隔（多少batch）
parser.add_argument('--log_interval', type=int, default=10,
                    help='interval (number of batches) of logging')
parser.add_argument('--device', type=str, default='cuda',
                    choices=['cuda', 'cpu'])

# 随机数种子
parser.add_argument('--seed', type=int, default=111, help='random seed')
parser.add_argument('--shuffle_nodes', action='store_true',
                    default=False, help='shuffle nodes for debugging')
parser.add_argument('-g', '--torch_geom', action='store_true', 
                    default=False, help='use PyTorch Geometric')
parser.add_argument('-a', '--adj_sq', action='store_true', default=False,
                    help='use A^2 instead of A as an adjacency matrix')
parser.add_argument('-s', '--scale_identity', action='store_true', default=False,
                    help='use 2I instead of I for self connections')
parser.add_argument('-v', '--visualize', action='store_true', default=False,
                    help='only for unet: save some adjacency matrices and other data as images')
parser.add_argument('-c', '--use_cont_node_attr', action='store_true', default=False,
                    help='use continuous node attributes in addition to discrete ones')

args = parser.parse_args()

'''
# 正式设置参数
# -D ENZYMES -f 128,128,128 --n_hidden 256 --lr 0.0005 --epochs 100 --lr_decay_step 150 -g 
# python graph_unet.py -D ENZYMES -M mgcn -K 4 -f 32,64,512 --n_hidden 256 --n_hidden_edge 128 --bn --lr 0.001 --epochs 50 --lr_decay_steps 25,35,45 -g
args.dataset = 'ENZYMES'  #ENZYMES
args.model = 'gcn'
args.filter_scale = 4
args.filters = '32,64,512'
args.n_hidden = 256
args.n_hidden_edge = 128
args.bn = True
args.lr = 0.001
args.epochs = 50
args.lr_decay_steps = '25,35,45'
args.torch_geom = True 
args.use_cont_node_attr = True
args.device = 'cpu'
'''


# Experiment parameters
args.batch_size = 32
args.threads = 0
args.lr = 0.005
args.epochs = 40
args.log_interval = 10
args.wd = 1e-4
args.dataset = 'BEHAVIOR'    # PROTEINS  BEHAVIOR
args.model = 'gcn'  # 'gcn', 'unet'
args.device = 'cpu'  # 'cuda', 'cpu'
args.visualize = True
args.shuffle_nodes = False
args.folds = 10  # 10-fold cross validation
args.seed = 111
args.torch_geom = False
args.use_cont_node_attr = True

if args.torch_geom: #PyTorch geometric是一个基于pytorch的图网络处理库
    from torch_geometric.datasets import TUDataset
    import torch_geometric.transforms as T

args.filters = list(map(int, args.filters.split(',')))
args.lr_decay_steps = list(map(int, args.lr_decay_steps.split(',')))

for arg in vars(args):
    print(arg, getattr(args, arg))

# train,val,test splits for COLORS and TRIANGLES and 10-fold cross validation for other datasets
n_folds = args.folds
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
rnd_state = np.random.RandomState(args.seed)


def split_ids(ids, folds=3): # 10折验证，返回每轮次训练集和测试集编号

    if args.dataset == 'COLORS-3':
        assert folds == 1, 'this dataset has train, val and test splits'
        train_ids = [np.arange(500)]
        val_ids = [np.arange(500, 3000)]
        test_ids = [np.arange(3000, 10500)]
    elif args.dataset == 'TRIANGLES':
        assert folds == 1, 'this dataset has train, val and test splits'
        train_ids = [np.arange(30000)]
        val_ids = [np.arange(30000, 35000)]
        test_ids = [np.arange(35000, 45000)]
    else:
        n = len(ids) #样本总数（要进行k-folds验证，因此每个group中的样本数至少为 n / folds）
        stride = int(np.ceil(n / float(folds))) #取样本的步长（每个group的样本数，这里将步长略微取大了，为了保证没有样本会被遗漏)
        test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
        assert np.all( #assert断言，条件不满直接触发异常，不用等程序运行后出现崩溃
            np.unique(np.concatenate(test_ids)) == sorted(ids)), 'some graphs are missing in the test sets'
        assert len(test_ids) == folds, 'invalid test sets'
        train_ids = []
        for fold in range(folds): #对于10-fold交叉验证，将样本集分成10份。每一轮迭代(共十轮)的过程中，使用其中一个group作为testdata,剩下的九个group作为traindata。
            train_ids.append(
                np.array([e for e in ids if e not in test_ids[fold]]))
            assert len(train_ids[fold]) + len(test_ids[fold]) == len(
                np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'
    # 这里返回的test_ids就是十轮迭代的testdata，train_ids是与其对应的traindata
    return train_ids, test_ids


# 若没有torch_geometric，则走这个，但我有，所以这里不走
if not args.torch_geom:
    # Unversal data loader and reader (can be used for other graph datasets from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)
    class GraphData(torch.utils.data.Dataset):
        def __init__(self,
                     datareader,
                     fold_id,
                     split):
            self.fold_id = fold_id
            self.split = split
            self.rnd_state = datareader.rnd_state
            self.set_fold(datareader.data, fold_id)

        def set_fold(self, data, fold_id):
            self.total = len(data['targets'])
            self.N_nodes_max = data['N_nodes_max']
            self.num_classes = data['num_classes']
            self.num_features = data['num_features']
            self.idx = data['splits'][fold_id][self.split]
            # use deepcopy to make sure we don't alter objects in folds
            self.labels = copy.deepcopy([data['targets'][i] for i in self.idx])
            self.adj_list = copy.deepcopy(
                [data['adj_list'][i] for i in self.idx])
            self.features_onehot = copy.deepcopy(
                [data['features_onehot'][i] for i in self.idx])
            print('%s: %d/%d' % (self.split.upper(),
                  len(self.labels), len(data['targets'])))

        def __len__(self):
            return len(self.labels)
        #这个函数很重要。后面训练模型过程中使用  for batch_idx, data in enumerate(loaders[0]) 得到的data就是该函数定义的形式(为什么实际数据中会多出两维？)
        def __getitem__(self, index):
            # convert to torch
            return [torch.from_numpy(self.features_onehot[index]).float(),  # node_features
                    # adjacency matrix
                    torch.from_numpy(self.adj_list[index]).float(),
                    int(self.labels[index])] #mask

    class DataReader():
        '''
        Class to read the txt files containing all data of the dataset.
        Should work for any dataset from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
        '''

        def __init__(self,
                     data_dir,  # folder with txt files
                     rnd_state=None,
                     use_cont_node_attr=True, # False,把该变量调整为True 
                     # use or not additional float valued node attributes available in some datasets
                     folds=10):

            self.data_dir = data_dir
            # 看成随机数种子
            self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
            self.use_cont_node_attr = use_cont_node_attr
            files = os.listdir(self.data_dir)
            print(files)
            data = {}
            # nodes: 字典{node_id:graph_id}， graphs: 字典{graph_id:[graph_nodes_id_list]}
            nodes, graphs = self.read_graph_nodes_relations( #graph_indicator 中对应的是每个节点所属的graph_id
                list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0]) #filter(function, iterable) 用于过滤序列，过滤掉不符合条件的元素
            
            # data['adj_list'] 是所有图的邻接矩阵adj_list。 adj_list[graph_id][x,y] 表示第 graph_id张图中x与y是否连接。 
            # 注意：这里得到的adj_list中的邻接矩阵大小是不一样的，因为每张图中节点数目不一样。后面需要将每个batch中的邻接矩阵的形状保持一致。
            data['adj_list'] = self.read_graph_adj(
                list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)
            
            node_labels_file = list(
                filter(lambda f: f.find('node_labels') >= 0, files))
            if len(node_labels_file) == 1:
                data['features'] = self.read_node_features(
                    node_labels_file[0], nodes, graphs, fn=lambda s: int(s.strip()))
            else:
                data['features'] = None

            data['targets'] = np.array(
                self.parse_txt_file(list(filter(lambda f: f.find('graph_labels') >= 0 or f.find('graph_attributes') >= 0, files))[0],
                                    line_parse_fn=lambda s: int(float(s.strip()))))

            if self.use_cont_node_attr:
                data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0],
                                                       nodes, graphs,
                                                       fn=lambda s: np.array(list(map(float, s.strip().split(',')))))

            features, n_edges, degrees = [], [], []
            for sample_id, adj in enumerate(data['adj_list']):
                N = len(adj)  # number of nodes
                if data['features'] is not None:
                    assert N == len(data['features'][sample_id]), (N, len(
                        data['features'][sample_id]))
                if not np.allclose(adj, adj.T):
                    print(sample_id, 'not symmetric')
                n = np.sum(adj)  # total sum of edges
                assert n % 2 == 0, n
                # undirected edges, so need to divide by 2
                n_edges.append(int(n / 2))
                degrees.extend(list(np.sum(adj, 1)))
                if data['features'] is not None:
                    features.append(np.array(data['features'][sample_id]))

            # Create features over graphs as one-hot vectors for each node
            if data['features'] is not None:
                features_all = np.concatenate(features)
                features_min = features_all.min()
                # number of possible values
                num_features = int(features_all.max() - features_min + 1)

            max_degree = np.max(degrees)
            features_onehot = []
            for sample_id, adj in enumerate(data['adj_list']):
                N = adj.shape[0]
                if data['features'] is not None:
                    x = data['features'][sample_id]
                    feature_onehot = np.zeros((len(x), num_features))
                    for node, value in enumerate(x):
                        feature_onehot[node, value - features_min] = 1
                else:
                    feature_onehot = np.empty((N, 0))
                if self.use_cont_node_attr:
                    if args.dataset in ['COLORS-3', 'TRIANGLES']:
                        # first column corresponds to node attention and shouldn't be used as node features
                        feature_attr = np.array(data['attr'][sample_id])[:, 1:]
                    else:
                        feature_attr = np.array(data['attr'][sample_id])
                else:
                    feature_attr = np.empty((N, 0))
                if args.degree:
                    degree_onehot = np.zeros((N, max_degree + 1))
                    degree_onehot[np.arange(N), np.sum(
                        adj, 1).astype(np.int32)] = 1
                else:
                    degree_onehot = np.empty((N, 0))
                # 这里
                node_features = np.concatenate(
                    (feature_onehot, feature_attr, degree_onehot), axis=1)
                if node_features.shape[1] == 0:
                    # dummy features for datasets without node labels/attributes
                    # node degree features can be used instead
                    node_features = np.ones((N, 1))
                features_onehot.append(node_features)

            num_features = features_onehot[0].shape[1]

            shapes = [len(adj) for adj in data['adj_list']]
            labels = data['targets']  # graph class labels
            labels -= np.min(labels)  # to start from 0

            classes = np.unique(labels)
            num_classes = len(classes)

            if not np.all(np.diff(classes) == 1):
                print('making labels sequential, otherwise pytorch might crash')
                labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
                for lbl in range(num_classes):
                    labels_new[labels == classes[lbl]] = lbl
                labels = labels_new
                classes = np.unique(labels)
                assert len(np.unique(labels)) == num_classes, np.unique(labels)

            def stats(x):
                return (np.mean(x), np.std(x), np.min(x), np.max(x))

            print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(shapes)) 
            print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(n_edges))
            print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' %
                  stats(degrees))
            print('Node features dim: \t\t%d' % num_features)
            print('N classes: \t\t\t%d' % num_classes)
            print('Classes: \t\t\t%s' % str(classes))
            for lbl in classes:
                print('Class %d: \t\t\t%d samples' %
                      (lbl, np.sum(labels == lbl)))

            if data['features'] is not None:
                for u in np.unique(features_all):
                    print('feature {}, count {}/{}'.format(u,
                          np.count_nonzero(features_all == u), len(features_all)))

            N_graphs = len(labels)  # number of samples (graphs) in data
            assert N_graphs == len(data['adj_list']) == len(
                features_onehot), 'invalid data'

            # Create train/test sets first
            train_ids, test_ids = split_ids(
                rnd_state.permutation(N_graphs), folds=folds) 
            #N_graphs是图(samples)的个数，pernutation的功能是生成从 0~N_graphs的所有整数组成的随机数组(也就是将这些数随机打乱)
            # Create train sets
            splits = []
            for fold in range(len(train_ids)):
                splits.append({'train': train_ids[fold],
                               'test': test_ids[fold]})

            data['features_onehot'] = features_onehot
            data['targets'] = labels
            data['splits'] = splits
            data['N_nodes_max'] = np.max(shapes)  # max number of nodes
            data['num_features'] = num_features
            data['num_classes'] = num_classes

            self.data = data

        def parse_txt_file(self, fpath, line_parse_fn=None):
            with open(pjoin(self.data_dir, fpath), 'r') as f:
                lines = f.readlines()
            data = [line_parse_fn(
                s) if line_parse_fn is not None else s for s in lines]
            return data

        def read_graph_adj(self, fpath, nodes, graphs):
            edges = self.parse_txt_file(
                fpath, line_parse_fn=lambda s: s.split(','))
            adj_dict = {}
            for edge in edges:
                # -1 because of zero-indexing in our code
                node1 = int(edge[0].strip()) - 1
                node2 = int(edge[1].strip()) - 1
                graph_id = nodes[node1]
                assert graph_id == nodes[node2], ('invalid data',
                                                  graph_id, nodes[node2])
                if graph_id not in adj_dict:
                    n = len(graphs[graph_id]) #这里n是当前图中节点个数
                    adj_dict[graph_id] = np.zeros((n, n))
                ind1 = np.where(graphs[graph_id] == node1)[0]
                ind2 = np.where(graphs[graph_id] == node2)[0]
                assert len(ind1) == len(ind2) == 1, (ind1, ind2)
                adj_dict[graph_id][ind1, ind2] = 1

            adj_list = [adj_dict[graph_id]
                        for graph_id in sorted(list(graphs.keys()))]

            return adj_list

        def read_graph_nodes_relations(self, fpath):
            graph_ids = self.parse_txt_file(
                fpath, line_parse_fn=lambda s: int(s.rstrip()))
            nodes, graphs = {}, {} #两个都是字典型数据 nodes存储{node_id:graph_id}， graphs存储{graph_id:[graph_nodes_id_list]}
            for node_id, graph_id in enumerate(graph_ids):
                if graph_id not in graphs:
                    graphs[graph_id] = []
                graphs[graph_id].append(node_id)
                nodes[node_id] = graph_id
            graph_ids = np.unique(list(graphs.keys()))
            for graph_id in graph_ids:
                graphs[graph_id] = np.array(graphs[graph_id]) # 把graphs中的list型数据转换成array
            return nodes, graphs

        def read_node_features(self, fpath, nodes, graphs, fn):
            node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
            node_features = {}
            for node_id, x in enumerate(node_features_all):
                graph_id = nodes[node_id]
                if graph_id not in node_features:
                    node_features[graph_id] = [None] * len(graphs[graph_id])
                ind = np.where(graphs[graph_id] == node_id)[0]
                assert len(ind) == 1, ind
                assert node_features[graph_id][ind[0]
                                               ] is None, node_features[graph_id][ind[0]]
                node_features[graph_id][ind[0]] = x
            node_features_lst = [node_features[graph_id]
                                 for graph_id in sorted(list(graphs.keys()))]
            return node_features_lst


# NN layers and models
class GraphConv(nn.Module):
    '''
    Graph Convolution Layer according to (T. Kipf and M. Welling, ICLR 2017) if K<=1
    Chebyshev Graph Convolution Layer according to (M. Defferrard, X. Bresson, and P. Vandergheynst, NIPS 2017) if K>1
    Additional tricks (power of adjacency matrix and weighted self connections) as in the Graph U-Net paper
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 # number of relation types (adjacency matrices)
                 n_relations=1,
                 K=1,  # GCN is K<=1, else ChebNet
                 activation=None,
                 bnorm=False,
                 adj_sq=False,
                 scale_identity=False):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features * K *
                            n_relations, out_features=out_features)
        self.n_relations = n_relations
        assert K > 0, ('filter scale must be greater than 0', K)
        self.K = K
        self.activation = activation
        self.bnorm = bnorm
        if self.bnorm:
            self.bn = nn.BatchNorm1d(out_features)
        self.adj_sq = adj_sq
        self.scale_identity = scale_identity

    def chebyshev_basis(self, L, X, K):
        if K > 1:
            Xt = [X]
            Xt.append(torch.bmm(L, X))  # B,N,F
            for k in range(2, K):
                Xt.append(2 * torch.bmm(L, Xt[k - 1]) - Xt[k - 2])  # B,N,F
            Xt = torch.cat(Xt, dim=2)  # B,N,K,F
            return Xt
        else:
            # GCN
            assert K == 1, K
            return torch.bmm(L, X)  # B,N,1,F

    def laplacian_batch(self, A):
        batch, N = A.shape[:2]
        if self.adj_sq:
            A = torch.bmm(A, A)  # use A^2 to increase graph connectivity
        A_hat = A
        if self.K < 2 or self.scale_identity:
            I = torch.eye(N).unsqueeze(0).to(args.device)
            if self.scale_identity:
                I = 2 * I  # increase weight of self connections
            if self.K < 2:
                A_hat = A + I
        D_hat = (torch.sum(A_hat, 1) + 1e-5) ** (-0.5)
        L = D_hat.view(batch, N, 1) * A_hat * D_hat.view(batch, 1, N)
        return L

    def forward(self, data):
        x, A, mask = data[:3] #mask是干什么的？
        # print('in', x.shape, torch.sum(torch.abs(torch.sum(x, 2)) > 0))
        if len(A.shape) == 3:
            A = A.unsqueeze(3)
        x_hat = []

        for rel in range(self.n_relations):
            L = self.laplacian_batch(A[:, :, :, rel])
            x_hat.append(self.chebyshev_basis(L, x, self.K))
        x = self.fc(torch.cat(x_hat, 2))

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(2)

        x = x * mask  # to make values of dummy nodes zeros again, otherwise the bias is added after applying self.fc which affects node embeddings in the following layers

        if self.bnorm:
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        if self.activation is not None:
            x = self.activation(x)
        return (x, A, mask)


class GCN(nn.Module):
    '''
    Baseline Graph Convolutional Network with a stack of Graph Convolution Layers and global pooling over nodes.
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 64, 64],
                 K=1,
                 bnorm=False,
                 n_hidden=0,
                 dropout=0.2,
                 adj_sq=False,
                 scale_identity=False):
        super(GCN, self).__init__()

        # Graph convolution layers
        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f,
                                                K=K,
                                                activation=nn.ReLU(
                                                    inplace=True),
                                                bnorm=bnorm,
                                                adj_sq=adj_sq,
                                                scale_identity=scale_identity) for layer, f in enumerate(filters)]))

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            fc.append(nn.ReLU(inplace=True))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    # 对某个Filter抽取到若干特征值,只取得其中最大的那个Pooling层作为保留值
    # 其他特征值全部抛弃,值最大代表只保留这些特征中最强的,抛弃其他弱的此类特征
    def forward(self, data):
        x = self.gconv(data)[0]

        # max pooling over nodes (usually performs better than average)
        x = torch.max(x, dim=1)[0].squeeze()
        x = self.fc(x)
        return x


class GraphUnet(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 filters=[64, 64, 64],
                 K=1,
                 bnorm=False,
                 n_hidden=0,
                 dropout=0.2,
                 adj_sq=False,
                 scale_identity=False,
                 shuffle_nodes=False,
                 visualize=False,
                 pooling_ratios=[0.8, 0.8]):
        super(GraphUnet, self).__init__()

        self.shuffle_nodes = shuffle_nodes
        self.visualize = visualize
        self.pooling_ratios = pooling_ratios
        # Graph convolution layers
        self.gconv = nn.ModuleList([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                              out_features=f,
                                              K=K,
                                              activation=nn.ReLU(inplace=True),
                                              bnorm=bnorm,
                                              adj_sq=adj_sq,
                                              scale_identity=scale_identity) for layer, f in enumerate(filters)])
        # Pooling layers
        self.proj = []
        for layer, f in enumerate(filters[:-1]):
            # Initialize projection vectors similar to weight/bias initialization in nn.Linear
            fan_in = filters[layer]
            p = Parameter(torch.Tensor(fan_in, 1))
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(p, -bound, bound)
            self.proj.append(p)
        self.proj = nn.ParameterList(self.proj)

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        # data: [node_features, A, graph_support, N_nodes, label]
        if self.shuffle_nodes:
            # shuffle nodes to make sure that the model does not adapt to nodes order (happens in some cases)
            N = data[0].shape[1]
            idx = torch.randperm(N)
            data = (data[0][:, idx], data[1][:, idx, :]
                    [:, :, idx], data[2][:, idx], data[3])

        sample_id_vis, N_nodes_vis = -1, -1
        for layer, gconv in enumerate(self.gconv):
            N_nodes = data[3]

            # TODO: remove dummy or dropped nodes for speeding up forward/backward passes
            # data = (data[0][:, :N_nodes_max], data[1][:, :N_nodes_max, :N_nodes_max], data[2][:, :N_nodes_max], data[3])

            x, A = data[:2]

            B, N, _ = x.shape

            # visualize data
            if self.visualize and layer < len(self.gconv) - 1:
                for b in range(B):
                    if (layer == 0 and N_nodes[b] < 20 and N_nodes[b] > 10) or sample_id_vis > -1:
                        if sample_id_vis > -1 and sample_id_vis != b:
                            continue
                        if N_nodes_vis < 0:
                            N_nodes_vis = N_nodes[b]
                        plt.figure()
                        plt.imshow(
                            A[b][:N_nodes_vis, :N_nodes_vis].data.cpu().numpy())
                        plt.title('layer %d, Input adjacency matrix' % (layer))
                        plt.savefig('input_adjacency_%d.png' % layer)
                        sample_id_vis = b
                        break

            # clone as we are going to make inplace changes
            mask = data[2].clone()
            x = gconv(data)[0]  # graph convolution
            if layer < len(self.gconv) - 1:
                B, N, C = x.shape
                # project features
                y = torch.mm(x.view(B * N, C), self.proj[layer]).view(B, N)
                # node scores used for ranking below
                y = y / (torch.sum(self.proj[layer] ** 2).view(1, 1) ** 0.5)
                # get indices of y values in the ascending order
                idx = torch.sort(y, dim=1)[1]
                # number of removed nodes
                N_remove = (N_nodes.float() *
                            (1 - self.pooling_ratios[layer])).long()

                # sanity checks
                assert torch.all(
                    N_nodes > N_remove), 'the number of removed nodes must be large than the number of nodes'
                for b in range(B):
                    # check that mask corresponds to the actual (non-dummy) nodes
                    assert torch.sum(mask[b]) == float(
                        N_nodes[b]), (torch.sum(mask[b]), N_nodes[b])

                N_nodes_prev = N_nodes
                N_nodes = N_nodes - N_remove

                for b in range(B):
                    # take indices of non-dummy nodes for current data example
                    idx_b = idx[b, mask[b, idx[b]] == 1]
                    assert len(idx_b) >= N_nodes[b], (
                        len(idx_b), N_nodes[b])  # number of indices must be at least as the number of nodes
                    # set mask values corresponding to the smallest y-values to 0
                    mask[b, idx_b[:N_remove[b]]] = 0

                # sanity checks
                for b in range(B):
                    # check that the new mask corresponds to the actual (non-dummy) nodes
                    assert torch.sum(mask[b]) == float(N_nodes[b]), (
                        b, torch.sum(mask[b]), N_nodes[b], N_remove[b], N_nodes_prev[b])
                    # make sure that y-values of selected nodes are larger than of dropped nodes
                    s = torch.sum(y[b] >= torch.min((y * mask.float())[b]))
                    assert s >= float(
                        N_nodes[b]), (s, N_nodes[b], (y * mask.float())[b])

                mask = mask.unsqueeze(2)
                # propagate only part of nodes using the mask
                x = x * torch.tanh(y).unsqueeze(2) * mask
                A = mask * A * mask.view(B, 1, N)
                mask = mask.squeeze()
                data = (x, A, mask, N_nodes)

                # visualize data
                if self.visualize and sample_id_vis > -1:
                    b = sample_id_vis
                    plt.figure()
                    plt.imshow(y[b].view(N, 1).expand(N, 2)[
                               :N_nodes_vis].data.cpu().numpy())
                    plt.title('Node ranking')
                    plt.colorbar()
                    plt.savefig('nodes_ranking_%d.png' % layer)
                    plt.figure()
                    plt.imshow(mask[b].view(N, 1).expand(N, 2)[
                               :N_nodes_vis].data.cpu().numpy())
                    plt.title('Pooled nodes (%d/%d)' %
                              (mask[b].sum(), N_nodes_prev[b]))
                    plt.savefig('pooled_nodes_mask_%d.png' % layer)
                    plt.figure()
                    plt.imshow(
                        A[b][:N_nodes_vis, :N_nodes_vis].data.cpu().numpy())
                    plt.title('Pooled adjacency matrix')
                    plt.savefig('pooled_adjacency_%d.png' % layer)
                    print('layer %d: visualizations saved ' % layer)

        if self.visualize and sample_id_vis > -1:
            self.visualize = False  # to prevent visualization for the following batches

        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes
        x = self.fc(x)
        return x


class MGCN(nn.Module):
    '''
    Multigraph Convolutional Network based on (B. Knyazev et al., "Spectral Multigraph Networks for Discovering and Fusing Relationships in Molecules")
    '''

    def __init__(self,
                 in_features,
                 out_features,
                 n_relations,
                 filters=[64, 64, 64],
                 K=1,
                 bnorm=False,
                 n_hidden=0,
                 n_hidden_edge=32,
                 dropout=0.2,
                 adj_sq=False,
                 scale_identity=False):
        super(MGCN, self).__init__()

        # Graph convolution layers
        self.gconv = nn.Sequential(*([GraphConv(in_features=in_features if layer == 0 else filters[layer - 1],
                                                out_features=f,
                                                n_relations=n_relations,
                                                K=K,
                                                activation=nn.ReLU(
                                                    inplace=True),
                                                bnorm=bnorm,
                                                adj_sq=adj_sq,
                                                scale_identity=scale_identity) for layer, f in enumerate(filters)]))

        # Edge prediction NN
        self.edge_pred = nn.Sequential(nn.Linear(in_features * 2, n_hidden_edge),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(n_hidden_edge, 1))

        # Fully connected layers
        fc = []
        if dropout > 0:
            fc.append(nn.Dropout(p=dropout))
        if n_hidden > 0:
            fc.append(nn.Linear(filters[-1], n_hidden))
            if dropout > 0:
                fc.append(nn.Dropout(p=dropout))
            n_last = n_hidden
        else:
            n_last = filters[-1]
        fc.append(nn.Linear(n_last, out_features))
        self.fc = nn.Sequential(*fc)

    def forward(self, data):
        # data: [node_features, A, graph_support, N_nodes, label]

        # Predict edges based on features
        x = data[0]
        B, N, C = x.shape
        mask = data[2]
        # find indices of nodes
        x_cat, idx = [], []
        for b in range(B):
            n = int(mask[b].sum())
            node_i = torch.nonzero(mask[b]).repeat(1, n).view(-1, 1)
            node_j = torch.nonzero(mask[b]).repeat(n, 1).view(-1, 1)
            # skip loops and symmetric connections
            triu = (node_i < node_j).squeeze()
            x_cat.append(torch.cat((x[b, node_i[triu]], x[b, node_j[triu]]), 2).view(
                int(torch.sum(triu)), C * 2))
            idx.append((node_i * N + node_j)[triu].squeeze())

        x_cat = torch.cat(x_cat)
        idx_flip = np.concatenate((np.arange(C, 2 * C), np.arange(C)))
        # predict values and encourage invariance to nodes order
        y = torch.exp(0.5 * (self.edge_pred(x_cat) +
                      self.edge_pred(x_cat[:, idx_flip])).squeeze())
        A_pred = torch.zeros(B, N * N, device=args.device)
        c = 0
        for b in range(B):
            A_pred[b, idx[b]] = y[c:c + idx[b].nelement()]
            c += idx[b].nelement()
        A_pred = A_pred.view(B, N, N)
        A_pred = (A_pred + A_pred.permute(0, 2, 1))  # assume undirected edges

        # Use both annotated and predicted adjacency matrices to learn a GCN
        data = (x, torch.stack((data[1], A_pred), 3), mask)
        x = self.gconv(data)[0]
        x = torch.max(x, dim=1)[0].squeeze()  # max pooling over nodes
        x = self.fc(x)
        return x
'''
collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
'''

# 整理batch，一批batch中每个图的矩阵规模相同（以该批中最大的那个为准，不够补0）
# 10-folds validation 完整训练10轮，1轮50epoch，每个epoch分batch训练，一个batch32个训练数据
def collate_batch(batch):
    '''
    Creates a batch of same size graphs by zero-padding node features and adjacency matrices up to
    the maximum number of nodes in the CURRENT batch rather than in the entire dataset.
    Graphs in the batches are usually much smaller than the largest graph in the dataset, so this method is fast.
    :param batch: batch in the PyTorch Geometric format or [node_features*batch_size, A*batch_size, label*batch_size]
    :return: [node_features, A, graph_support, N_nodes, label]
    '''
    B = len(batch) #一个batch中有args.batch_size个样本(如果DataLoader中设置suffle=true，则是随机选择的batch_size个样本)
    if args.torch_geom:
        N_nodes = [len(batch[b].x) for b in range(B)] #这是batch中每个图中节点个数组成的list
        C = batch[0].x.shape[1] # 节点的特征数量
    else:
        N_nodes = [len(batch[b][1]) for b in range(B)]
        C = batch[0][0].shape[1]
    N_nodes_max = int(np.max(N_nodes)) # Max Pooling

    
    graph_support = torch.zeros(B, N_nodes_max) #这是个batchSize*N_nodes_max的tensor
    A = torch.zeros(B, N_nodes_max, N_nodes_max) #这是一个batch的所有邻接矩阵的tensor
    x = torch.zeros(B, N_nodes_max, C) #这是batch中每个图每个节点的初始向量组成的tensor
    
    # zero-padding
    for b in range(B):
        if args.torch_geom:
            x[b, :N_nodes[b]] = batch[b].x
            A[b].index_put_(
                (batch[b].edge_index[0], batch[b].edge_index[1]), torch.Tensor([1]))
        else:
            x[b, :N_nodes[b]] = batch[b][0]
            A[b, :N_nodes[b], :N_nodes[b]] = batch[b][1]
        # mask with values of 0 for dummy (zero padded) nodes, otherwise 1
        graph_support[b][:N_nodes[b]] = 1

    N_nodes = torch.from_numpy(np.array(N_nodes)).long() #把N_nodes转成tensor
    labels = torch.from_numpy(np.array( # batch的第三个标签是label
         [batch[b].y.item() if args.torch_geom else batch[b][2] for b in range(B)])).long()
    #    [batch[b].y if args.torch_geom else batch[b][2] for b in range(B)])).long()
    return [x, A, graph_support, N_nodes, labels]


# other datasets can be for the regression task (see their README.txt)
is_regression = args.dataset in ['COLORS-3', 'TRIANGLES']
transforms = []  # for PyTorch Geometric
if args.dataset in ['COLORS-3', 'TRIANGLES']:
    assert n_folds == 1, 'use train, val and test splits for these datasets'
    assert args.use_cont_node_attr, 'node attributes should be used for these datasets'

    if args.torch_geom:
        # Class to read note attention from DS_node_attributes.txt
        class HandleNodeAttention(object):
            def __call__(self, data):
                if args.dataset == 'COLORS-3':
                    data.attn = torch.softmax(data.x[:, 0], dim=0)
                    data.x = data.x[:, 1:]
                else:
                    data.attn = torch.softmax(data.x, dim=0)
                    data.x = None
                return data

        transforms.append(HandleNodeAttention())
else:
    assert n_folds == 10, '10-fold cross-validation should be used for other datasets'

print('Regression={}'.format(is_regression))
print('Loading data')

if is_regression:
    def loss_fn(output, target, reduction='mean'):
        loss = (target.float().squeeze() - output.squeeze()) ** 2
        return loss.sum() if reduction == 'sum' else loss.mean()

    def predict_fn(output): return output.round().long().detach().cpu()
else:
    loss_fn = F.cross_entropy

    def predict_fn(output): return output.max(
        1, keepdim=True)[1].detach().cpu()

if args.torch_geom:
    if args.degree:
        if args.dataset == 'TRIANGLES':
            max_degree = 14
        else:
            raise NotImplementedError('max_degree value should be specified in advance. '
                                      'Try running without --torch_geom (-g) and look at dataset statistics printed out by our code.')

    if args.degree:
        transforms.append(T.OneHotDegree(max_degree=max_degree, cat=False))

    dataset = TUDataset('./data/%s/' % args.dataset, name=args.dataset,
                        use_node_attr=args.use_cont_node_attr,
                        transform=T.Compose(transforms))
    train_ids, test_ids = split_ids(
        rnd_state.permutation(len(dataset)), folds=n_folds)

else:
    datareader = DataReader(data_dir='./data/%s/' % args.dataset,
                            rnd_state=rnd_state,
                            folds=n_folds,
                            use_cont_node_attr=args.use_cont_node_attr)

acc_folds = []
# k-flods需要进行k轮的数据训练
for fold_id in range(n_folds):

    loaders = []
    for split in ['train', 'test']: #定义两个loader，分别用于存储train_data和test_data
        if args.torch_geom:
            #gdata = dataset[torch.from_numpy(
            #    (train_ids if split.find('train') >= 0 else test_ids)[fold_id])]
            gdata = dataset[list((train_ids if split.find('train') >= 0 else test_ids)[fold_id])]  
        else:
            gdata = GraphData(fold_id=fold_id,
                              datareader=datareader,
                              split=split)

        loader = DataLoader(gdata,
                            batch_size=args.batch_size,
                            shuffle=split.find('train') >= 0,
                            num_workers=args.threads,
                            collate_fn=collate_batch)
        loaders.append(loader)

    print('\nFOLD {}/{}, train {}, test {}'.format(fold_id + 1,
          n_folds, len(loaders[0].dataset), len(loaders[1].dataset)))

    if args.model == 'gcn':
        model = GCN(in_features=loaders[0].dataset.num_features,
                    out_features=1 if is_regression else loaders[0].dataset.num_classes,
                    n_hidden=args.n_hidden,
                    filters=args.filters,
                    K=args.filter_scale,
                    bnorm=args.bn,
                    dropout=args.dropout,
                    adj_sq=args.adj_sq,
                    scale_identity=args.scale_identity).to(args.device)
    elif args.model == 'unet':
        model = GraphUnet(in_features=loaders[0].dataset.num_features,
                          out_features=1 if is_regression else loaders[0].dataset.num_classes,
                          n_hidden=args.n_hidden,
                          filters=args.filters,
                          K=args.filter_scale,
                          bnorm=args.bn,
                          dropout=args.dropout,
                          adj_sq=args.adj_sq,
                          scale_identity=args.scale_identity,
                          shuffle_nodes=args.shuffle_nodes,
                          visualize=args.visualize).to(args.device)
    elif args.model == 'mgcn':
        model = MGCN(in_features=loaders[0].dataset.num_features,
                     out_features=1 if is_regression else loaders[0].dataset.num_classes,
                     n_relations=2,
                     n_hidden=args.n_hidden,
                     n_hidden_edge=args.n_hidden_edge,
                     filters=args.filters,
                     K=args.filter_scale,
                     bnorm=args.bn,
                     dropout=args.dropout,
                     adj_sq=args.adj_sq,
                     scale_identity=args.scale_identity).to(args.device)

    else:
        raise NotImplementedError(args.model)

    print('\nInitialize model')
    print(model)
    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print('N trainable parameters:', np.sum([p.numel() for p in train_params]))

    optimizer = optim.Adam(train_params, lr=args.lr,
                           weight_decay=args.wd, betas=(0.5, 0.999))
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, args.lr_decay_steps, gamma=0.1)

    # Normalization of continuous node features
    # if args.use_cont_node_attr:
    #     x = []
    #     for batch_idx, data in enumerate(loaders[0]):
    #         if args.torch_geom:
    #             node_attr_dim = loaders[0].dataset.props['node_attr_dim']
    #         x.append(data[0][:, :, :node_attr_dim].view(-1, node_attr_dim).data)
    #     x = torch.cat(x)
    #     mn, sd = torch.mean(x, dim=0).to(args.device), torch.std(x, dim=0).to(args.device) + 1e-5
    #     print(mn, sd)
    # else:
    #     mn, sd = 0, 1

    # def norm_features(x):
    #     x[:, :, :node_attr_dim] = (x[:, :, :node_attr_dim] - mn) / sd

    def train(train_loader):
        scheduler.step()
        model.train()
        start = time.time()
        train_loss, n_samples = 0, 0
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)
            # if args.use_cont_node_attr:
            #     data[0] = norm_features(data[0])
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, data[4])
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)
            if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader) - 1:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
                    epoch + 1, n_samples, len(train_loader.dataset),
                    100. * (batch_idx + 1) /
                    len(train_loader), loss.item(), train_loss / n_samples,
                    time_iter / (batch_idx + 1)))

    def test(test_loader):
        model.eval()
        start = time.time()
        test_loss, correct, n_samples = 0, 0, 0
        for batch_idx, data in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)
            # if args.use_cont_node_attr:
            #     data[0] = norm_features(data[0])
            output = model(data)
            loss = loss_fn(output, data[4], reduction='sum')
            test_loss += loss.item()
            n_samples += len(output)
            pred = predict_fn(output)

            correct += pred.eq(data[4].detach().cpu().view_as(pred)
                               ).sum().item()

        acc = 100. * correct / n_samples
        print('Test set (epoch {}): Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%) \tsec/iter: {:.4f}\n'.format(
            epoch + 1,
            test_loss / n_samples,
            correct,
            n_samples,
            acc, (time.time() - start) / len(test_loader)))
        return acc

    for epoch in range(args.epochs):
        train(loaders[0])  # no need to evaluate after each epoch
    acc = test(loaders[1])
    acc_folds.append(acc)

print(acc_folds)
print('{}-fold cross validation avg acc (+- std): {} ({})'.format(n_folds,
      np.mean(acc_folds), np.std(acc_folds)))
