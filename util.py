import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam


def edge_in(graph, edge):
    return torch.any(torch.all(graph == edge, dim=1))


def diff(graph1, graph2):
    count = 0
    for edge in graph1:
        if not edge_in(graph2, edge):
            count += 1
    for edge in graph2:
        if not edge_in(graph1, edge):
            count += 1
    return count


def train(
    model,
    data,
    epochs=200,
    lr=0.005,
    weight_decay=5e-4,
    edge_index=None,
    edge_weight=None,
    manip_coeff=0,
):
    model.train()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    edge_index = data.edge_index if edge_index is None else edge_index
    edge_weight = (
        data.edge_weight
        if "edge_weight" in data and edge_weight is None
        else edge_weight
    )
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(data.x, edge_index, edge_weight)
        loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])
        loss.backward(retain_graph=True)
        optimizer.step()


def accuracy(pred, y, mask):
    return (pred.argmax(-1)[mask] == y[mask]).float().mean()


@torch.no_grad()
def test(model, data, mask=None, edge_index=None):
    edge_index = data.edge_index if edge_index is None else edge_index
    mask = data.test_mask if mask is None else mask
    model.eval()
    pred = model(data.x, edge_index, data.edge_weight)
    return round(float(accuracy(pred, data.y, mask)), 3)


# The metric in PRBCD is assumed to be best if lower (like a loss).
def metric(*args, **kwargs):
    return -accuracy(*args, **kwargs)


def split_inductive(
    labels, n_per_class=20, seed=None, balance_test=True, test_size=0.1
):
    """
    Randomly split the training data.

    Parameters
    ----------
    labels: array-like [num_nodes]
        The class labels
    n_per_class : int
        Number of samples per class
    balance_test: bool
        wether to balance the classes in the test set; if true, take 10% of all nodes as test set
    seed: int
        Seed

    Returns
    -------
    split_labeled: array-like [n_per_class * nc]
        The indices of the training nodes
    split_val: array-like [n_per_class * nc]
        The indices of the validation nodes
    split_test: array-like [n_per_class * nc]
        The indices of the test nodes
    split_unlabeled: array-like [num_nodes - 3*n_per_class * nc]
        The indices of the unlabeled nodes
    """
    if seed is not None:
        np.random.seed(seed)
    nc = labels.max() + 1
    if balance_test:
        # compute n_per_class
        bins = np.bincount(labels)
        n_test_per_class = np.ceil(test_size * bins)
    else:
        n_test_per_class = np.ones(nc) * n_per_class

    split_labeled, split_val, split_test = [], [], []
    for label in range(nc):
        perm = np.random.permutation((labels == label).nonzero()[0])
        split_labeled.append(perm[:n_per_class])
        split_val.append(perm[n_per_class : 2 * n_per_class])
        split_test.append(
            perm[
                2 * n_per_class : 2 * n_per_class + n_test_per_class[label].astype(int)
            ]
        )

    split_labeled = np.random.permutation(np.concatenate(split_labeled))
    split_val = np.random.permutation(np.concatenate(split_val))
    split_test = np.random.permutation(np.concatenate(split_test))

    assert split_labeled.shape[0] == split_val.shape[0] == n_per_class * nc

    split_unlabeled = np.setdiff1d(
        np.arange(len(labels)), np.concatenate((split_labeled, split_val, split_test))
    )

    return split_labeled, split_unlabeled, split_val, split_test


def wandb_log_list(wandb, lst, tag):
    for l in lst:
        wandb.log({tag: l})
