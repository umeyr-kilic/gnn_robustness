import torch
import numpy as np
from deeprobust.graph.data import Dataset, PtbDataset, PrePtbDataset
from deeprobust.graph.defense import GCN, GCNJaccard, RGCN
from deeprobust.graph.global_attack import Metattack
from deeprobust.graph.targeted_attack import Nettack
from scipy.sparse import csr_matrix
import os
import pickle
import matplotlib.pyplot as plt


# as
def net_attack():
    path = os.path.join(os.getcwd(), 'tmp/')
    data = Dataset(root=path, name='cora', setting='prognn')

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    # Load nettack attacked data

    perturbed_data = PrePtbDataset(root=path, name='cora',
                                   attack_method='nettack',
                                   ptb_rate=3.0)  # here ptb_rate means number of perturbation per nodes
    perturbed_adj = perturbed_data.adj
    idx_test = perturbed_data.target_nodes


    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    gcn = gcn.to('cpu')
    gcn.fit(features, adj, labels, idx_train, idx_val)  # train on clean graph with earlystopping
    acc_clean = gcn.test(idx_test)

    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    gcn = gcn.to('cpu')
    gcn.fit(features, perturbed_adj, labels, idx_train, idx_val)  # train on poisoned graph
    acc_perturbated_3 = gcn.test(idx_test)

    perturbed_data = PrePtbDataset(root=path, name='cora',
                                   attack_method='nettack',
                                   ptb_rate=5.0)  # here ptb_rate means number of perturbation per nodes
    perturbed_adj = perturbed_data.adj
    idx_test = perturbed_data.target_nodes

    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    gcn = gcn.to('cpu')
    gcn.fit(features, adj, labels, idx_train, idx_val)  # train on clean graph with earlystopping
    acc_clean = gcn.test(idx_test)

    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    gcn = gcn.to('cpu')
    gcn.fit(features, perturbed_adj, labels, idx_train, idx_val)  # train on poisoned graph
    acc_perturbated_5 = gcn.test(idx_test)

    # region plot
    labels = ["clean_graph", "perturbated_graph_3", "perturbated_graph_5"]
    accuracies = [acc_clean, acc_perturbated_3, acc_perturbated_5]
    accuracies = [acc * 100 for acc in accuracies]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    acc_bar = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='maroon')

    ax.set_ylabel('Accuracies')
    ax.set_title("Accuracy of Graph Models")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.legend()

    for rect in acc_bar:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height,2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    plt.show()


# as
def meta_attack():
    path = os.path.join(os.getcwd(), 'tmp/')
    data = Dataset(root=path, name='cora')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    idx_unlabeled = np.union1d(idx_val, idx_test)
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                    with_relu=False, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(features, adj, labels, idx_train)
    model = Metattack(model=surrogate, nnodes=adj.shape[0], feature_shape=features.shape, device=device)
    model = model.to(device)
    perturbations = int(0.05 * (adj.sum() // 2))
    acc_log = model.attack(features, adj, labels, idx_train, idx_unlabeled, perturbations, ll_constraint=False)
    modified_adj = model.modified_adj

    """
    dim = len(modified_adj)
    print(modified_adj)  # sparse tensor
    row = modified_adj[0].numpy()
    col = modified_adj[1].numpy()
    edge_num = len(row)
    data = np.ones(edge_num)
    modified_adj_mtx = csr_matrix((data, (row, col)), shape=(dim, dim))
    

    model = GCN(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device)
    model = model.to(device)
    model.fit(features, modified_adj_mtx, labels, idx_train)
    model.eval()
    output = model.test(idx_test)
    """

    print(acc_log)
    with open('meta_attack_accuracy.data', 'wb') as filehandle:
        pickle.dump(acc_log, filehandle)


# not as
def gcn_jaccard_defense():
    path = os.path.join(os.getcwd(), 'tmp/')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = Dataset(root=path, name='cora', setting='nettack')
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    # load pre-attacked graph by mettack
    perturbed_data = PrePtbDataset(root=path, name='cora', attack_method='meta', ptb_rate=0.25)
    perturbed_data = PrePtbDataset(root=path, name='cora', attack_method='nettack', ptb_rate=5.0)  # here ptb_rate means number of perturbation per nodes
    perturbed_adj = perturbed_data.adj
    # idx_test = perturbed_data.target_nodes

    # Set up defense model and test performance
    model = GCNJaccard(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device)
    model = model.to(device)
    model.fit(features, perturbed_adj, labels, idx_train)
    model.eval()
    output = model.test(idx_test)

    # Test on GCN
    model = GCN(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device)
    model = model.to(device)
    model.fit(features, perturbed_adj, labels, idx_train)
    model.eval()
    output = model.test(idx_test)


# as
def jaccard_defense():
    ptb_rate = 0.05  # 0.25
    path = os.path.join(os.getcwd(), 'tmp/')
    data = Dataset(root=path, name='cora', setting='prognn')

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    # Load meta attacked data
    perturbed_data = PrePtbDataset(root=path,
                                   name='cora',
                                   attack_method='meta',
                                   ptb_rate=ptb_rate)
    perturbed_adj = perturbed_data.adj

    # Load nettack attacked data
    """
    perturbed_data = PrePtbDataset(root=path, name='cora',
                                   attack_method='nettack',
                                   ptb_rate=3.0)  # here ptb_rate means number of perturbation per nodes
    perturbed_adj = perturbed_data.adj
    idx_test = perturbed_data.target_nodes
    """

    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    gcn = gcn.to('cpu')
    gcn.fit(features, adj, labels, idx_train, idx_val)  # train on clean graph with earlystopping
    acc_clean = gcn.test(idx_test)

    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    gcn = gcn.to('cpu')
    gcn.fit(features, perturbed_adj, labels, idx_train, idx_val)  # train on poisoned graph
    acc_perturbated = gcn.test(idx_test)

    model = GCNJaccard(nfeat=features.shape[1],
                       nhid=16,
                       nclass=labels.max().item() + 1,
                       dropout=0.5, device='cpu').to('cpu')
    model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=0.03)
    acc_jaccard = model.test(idx_test)

    model = RGCN(nnodes=perturbed_adj.shape[0], nfeat=features.shape[1],
                 nclass=labels.max() + 1, nhid=32, device='cpu')
    model.fit(features, perturbed_adj, labels, idx_train, idx_val,
              train_iters=200, verbose=True)
    acc_rgcn = model.test(idx_test)

    # region plot
    labels = ["clean_graph", "perturbated_graph", "Jaccard Defense", "RGCN Defense"]
    accuracies = [acc_clean, acc_perturbated, acc_jaccard, acc_rgcn]
    accuracies = [acc * 100 for acc in accuracies]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    acc_bar = ax.bar(x - width/2, accuracies, width, label='Accuracy', color='maroon')

    ax.set_ylabel('Accuracies')
    ax.set_title("Accuracy of Graph Models (Perturbation Rate: {}%)".format(ptb_rate * 100))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.legend()

    for rect in acc_bar:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height,2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    plt.show()


# as
def defense_against_net():
    ptb_rate = 5.0
    path = os.path.join(os.getcwd(), 'tmp/')
    data = Dataset(root=path, name='cora', setting='prognn')

    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    # Load nettack attacked data

    perturbed_data = PrePtbDataset(root=path, name='cora',
                                   attack_method='nettack',
                                   ptb_rate=ptb_rate)  # here ptb_rate means number of perturbation per nodes
    perturbed_adj = perturbed_data.adj
    idx_test = perturbed_data.target_nodes

    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    gcn = gcn.to('cpu')
    gcn.fit(features, adj, labels, idx_train, idx_val)  # train on clean graph with earlystopping
    acc_clean = gcn.test(idx_test)

    gcn = GCN(nfeat=features.shape[1],
              nhid=16,
              nclass=labels.max().item() + 1,
              dropout=0.5, device='cpu')
    gcn = gcn.to('cpu')
    gcn.fit(features, perturbed_adj, labels, idx_train, idx_val)  # train on poisoned graph
    acc_perturbated = gcn.test(idx_test)

    model = GCNJaccard(nfeat=features.shape[1],
                       nhid=16,
                       nclass=labels.max().item() + 1,
                       dropout=0.5, device='cpu').to('cpu')
    model.fit(features, perturbed_adj, labels, idx_train, idx_val, threshold=0.03)
    acc_jaccard = model.test(idx_test)

    model = RGCN(nnodes=perturbed_adj.shape[0], nfeat=features.shape[1],
                 nclass=labels.max() + 1, nhid=32, device='cpu')
    model.fit(features, perturbed_adj, labels, idx_train, idx_val,
              train_iters=200, verbose=True)
    acc_rgcn = model.test(idx_test)

    # region plot
    labels = ["clean_graph", "perturbated_graph", "Jaccard Defense", "RGCN Defense"]
    accuracies = [acc_clean, acc_perturbated, acc_jaccard, acc_rgcn]
    accuracies = [acc * 100 for acc in accuracies]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    acc_bar = ax.bar(x - width / 2, accuracies, width, label='Accuracy', color='maroon')

    ax.set_ylabel('Accuracies')
    ax.set_title("Accuracy of Graph Models (Perturbation Rate: {})".format(ptb_rate))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.legend()

    for rect in acc_bar:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    fig.tight_layout()
    plt.show()


# not as
def net_attack_obsolete():
    path = os.path.join(os.getcwd(), 'tmp/')
    data = Dataset(root=path, name='cora')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    model = GCN(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device)
    model = model.to(device)
    model.fit(features, adj, labels, idx_train)
    model.eval()
    output = model.test(idx_test)

    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    # Setup Attack Model
    target_node = 10
    model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device='cpu').to('cpu')
    model = model.to(device)
    # Attack
    model.attack(features, adj, labels, target_node, n_perturbations=5)
    modified_adj = model.modified_adj  # scipy sparse matrix
    modified_features = model.modified_features  # scipy sparse matrix

    model = GCN(nfeat=features.shape[1], nclass=labels.max() + 1, nhid=16, device=device)
    model = model.to(device)
    model.fit(modified_features, modified_adj, labels, idx_train)
    model.eval()
    output = model.test(idx_test)


# region utils
def plot_list():
    with open('meta_attack_accuracy_org.data', 'rb') as filehandle:
        # read the data as binary data stream
        acc_log = pickle.load(filehandle)

    acc_log = [acc * 100 for acc in acc_log]
    plt.plot(acc_log)
    plt.xlabel("Number of perturbations")
    plt.ylabel("Accuracy of the model")
    plt.title("Perturbation vs Accuracy")
    plt.show()

    perturbation_percentage = [100 * ind / 10000 for ind in range(len(acc_log))]
    plt.plot(perturbation_percentage, acc_log)
    plt.xlabel("Percentage of perturbation %")
    plt.ylabel("Accuracy of the model %")
    plt.title("Perturbation vs Accuracy")
    plt.show()
# endregion


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # net_attack()
    # meta_attack()
    # plot_list()
    # jaccard_defense()
    defense_against_net()



