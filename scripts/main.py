import argparse
import copy
import itertools
import os.path as osp
import sys
from typing import Optional, Tuple

import higher
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch import Tensor
from torch.optim import Adam
from torch_geometric.datasets import Planetoid, WikiCS, WikipediaNetwork
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, softmax

sys.path.insert(1, osp.join(sys.path[0], ".."))

from greatx.defense import JaccardPurification, SVDPurification
from ogb.nodeproppred import PygNodePropPredDataset
import pickle

import wandb

from attacks import *
from models import *
from util import *

# from datasets import *

parser = argparse.ArgumentParser(
    prog="prbcd_attack", description="Launches PRBCD evasion and poisoning attacks"
)

parser.add_argument("-n", "--seed", default=12345, type=int)
parser.add_argument("-s", "--block-size", type=int, nargs="*")
parser.add_argument(
    "-b", "--budget", type=float, nargs="*", help="Total budget used in the attack."
)
parser.add_argument("-r", "--reps", default=1, type=int, help="Number of runs.")
parser.add_argument("-w", "--wandb", default=None, type=str, help="Wandb project name.")
parser.add_argument(
    "-m", "--model", nargs="*", choices=get_model(), default="gcn", type=str
)
parser.add_argument(
    "-d",
    "--dataset",
    nargs="*",
    choices=["Cora", "CiteSeer", "PubMed", "WikiCS", "arxiv", "squirrel"],
    default="Cora",
    type=str,
)
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument(
    "-a",
    "--attack",
    nargs="*",
    choices=["evasion", "sequential", "joint", "poisoning"],
    type=str,
)
parser.add_argument("-l", "--learning-rate", nargs="*", default=2_000, type=int)
parser.add_argument("-p", "--poisoning-epochs", default=125, type=int)
parser.add_argument("-e", "--evasion-epochs", default=125, type=int)
parser.add_argument("-i", "--inner-evasion-epochs", default=125, type=int)
parser.add_argument("--defense", default=None, type=str, choices=["jaccard", "SVD"])
parser.add_argument("--inductive", action="store_true")
parser.add_argument("--evasion-reps", default=1, type=int)
parser.add_argument("--test-size", default=0.1, type=float)
parser.add_argument("--train-per-class", default=20, type=int)
parser.add_argument("--reg-coeff", default=0, type=float, nargs="*")
args = parser.parse_args()


def vprint(*arg):
    if args.verbose:
        print(*arg)


vprint("Setup:", args)

WANDB_PROJECT_NAME = args.wandb
DATA_DIR = "../data"

torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vprint("running on", device)

LOSS_TYPE = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]


# normal poisoning
class PoisoningPRBCDAttack(PRBCDAttack):
    def _forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, **kwargs
    ) -> Tensor:
        """Forward model."""
        self.model.reset_parameters()

        with torch.enable_grad():
            ped = copy.copy(data)
            ped.x, ped.edge_index, ped.edge_weight = x, edge_index, edge_weight
            train(self.model, ped, n_epochs, lr, weight_decay)

        self.model.eval()
        return self.model(x, edge_index, edge_weight)

    def _forward_and_gradient(
        self, x: Tensor, labels: Tensor, idx_attack: Optional[Tensor] = None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Forward and update edge weights."""
        if not self.block_edge_weight.requires_grad:
            self.block_edge_weight.requires_grad = True

        self.model.reset_parameters()

        self.model.train()
        opt = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        with higher.innerloop_ctx(self.model, opt) as (fmodel, diffopt):
            edge_index, edge_weight = self._get_modified_adj(
                self.edge_index,
                self.edge_weight,
                self.block_edge_index,
                self.block_edge_weight,
            )

            # Normalize only once (only relevant if model normalizes adj)
            if hasattr(fmodel, "norm"):
                edge_index, edge_weight = fmodel.norm(
                    edge_index,
                    edge_weight,
                    num_nodes=x.size(0),
                    add_self_loops=True,
                )

            for _ in range(n_epochs):
                if hasattr(fmodel, "norm"):
                    pred = fmodel.forward(x, edge_index, edge_weight, skip_norm=True)
                else:
                    pred = fmodel.forward(x, edge_index, edge_weight)
                loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])
                diffopt.step(loss)

            pred = fmodel(x, edge_index, edge_weight)
            loss = self.loss(pred, labels, idx_attack)
            gradient = torch.autograd.grad(loss, self.block_edge_weight)[0]

        # Clip gradient for stability:
        clip_norm = 0.5
        grad_len_sq = gradient.square().sum()
        if grad_len_sq > clip_norm:
            gradient *= clip_norm / grad_len_sq.sqrt()

        self.model.eval()

        return loss, gradient


class EvasionAttack(torch.nn.Module):
    coeffs = {
        "max_final_samples": 20,
        "max_trials_sampling": 20,
        "with_early_stopping": True,
        "eps": 1e-7,
    }

    def __init__(
        self,
        model: torch.nn.Module,
        block_size: int,
        epochs: int = 125,
        epochs_resampling: int = 100,
        loss: Optional[Union[str, LOSS_TYPE]] = "prob_margin",
        metric: Optional[Union[str, LOSS_TYPE]] = None,
        lr: float = 1_000,
        is_undirected: bool = True,
        log: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.model = model
        self.block_size = block_size
        self.epochs = epochs

        if isinstance(loss, str):
            if loss == "masked":
                self.loss = self._masked_cross_entropy
            elif loss == "margin":
                self.loss = partial(self._margin_loss, reduce="mean")
            elif loss == "prob_margin":
                self.loss = self._probability_margin_loss
            elif loss == "tanh_margin":
                self.loss = self._tanh_margin_loss
            else:
                raise ValueError(f"Unknown loss `{loss}`")
        else:
            self.loss = loss

        self.is_undirected = is_undirected
        self.log = log
        self.metric = metric or self.loss

        self.epochs_resampling = epochs_resampling
        self.lr = lr

        self.coeffs.update(kwargs)

    def attack(
        self,
        x: Tensor,
        edge_index: Tensor,
        labels: Tensor,
        budget: int,
        idx_attack: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Attack the predictions for the provided model and graph.

        A subset of predictions may be specified with :attr:`idx_attack`. The
        attack is allowed to flip (i.e. add or delete) :attr:`budget` edges and
        will return the strongest perturbation it can find. It returns both the
        resulting perturbed :attr:`edge_index` as well as the perturbations.

        Args:
            x (torch.Tensor): The node feature matrix.
            edge_index (torch.Tensor): The edge indices.
            labels (torch.Tensor): The labels.
            budget (int): The number of allowed perturbations (i.e.
                number of edges that are flipped at most).
            idx_attack (torch.Tensor, optional): Filter for predictions/labels.
                Shape and type must match that it can index :attr:`labels`
                and the model's predictions.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`torch.Tensor`, :class:`torch.Tensor`)
        """
        self.model.eval()

        self.device = x.device
        assert kwargs.get("edge_weight") is None
        # edge_weight = torch.ones(edge_index.size(1), device=self.device)
        assert self.edge_weight is not None
        self.edge_index = edge_index.cpu().clone()
        self.edge_weight = self.edge_weight.cpu().clone()
        self.num_nodes = x.size(0)

        # For collecting attack statistics
        self.attack_statistics = defaultdict(list)

        # Prepare attack and define `self.iterable` to iterate over
        step_sequence = self._prepare(budget)

        for step in tqdm(
            step_sequence, disable=not self.log, desc="Evasion", leave=False
        ):
            loss, gradient = self._forward_and_gradient(x, labels, idx_attack, **kwargs)

            scalars = self._update(
                step, gradient, x, labels, budget, idx_attack, **kwargs
            )

            scalars["loss"] = loss.item()
            self._append_statistics(scalars)
        else:
            # get final relaxed adj matrix
            rel_edge_index, rel_edge_weight = self._get_modified_adj(
                self.edge_index,
                self.edge_weight,
                self.block_edge_index,
                self.block_edge_weight,
            )

        perturbed_edge_index, flipped_edges, edge_weight = self._close(
            x, labels, budget, idx_attack, **kwargs
        )

        assert flipped_edges.size(1) <= budget, (
            f"# perturbed edges {flipped_edges.size(1)} " f"exceeds budget {budget}"
        )

        return (
            perturbed_edge_index,
            flipped_edges,
            edge_weight,
            rel_edge_index,
            rel_edge_weight,
        )

    def _prepare(self, budget: int) -> Iterable[int]:
        """Prepare attack."""
        if self.block_size <= budget:
            raise ValueError(
                f"The search space size ({self.block_size}) must be "
                f"greater than the number of permutations ({budget})"
            )

        # For early stopping (not explicitly covered by pseudo code)
        self.best_metric = float("-Inf")

        # Sample initial search space (Algorithm 1, line 3-4)
        self._sample_random_block(budget)

        steps = range(self.epochs)
        return steps

    # @torch.no_grad()
    def _update(
        self,
        epoch: int,
        gradient: Tensor,
        x: Tensor,
        labels: Tensor,
        budget: int,
        idx_attack: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Update edge weights given gradient."""
        # Gradient update step (Algorithm 1, line 7)
        self.block_edge_weight = self._update_edge_weights(
            budget, self.block_edge_weight, epoch, gradient
        )

        # For monitoring
        pmass_update = torch.clamp(self.block_edge_weight, 0, 1)
        # Projection to stay within relaxed `L_0` budget
        # (Algorithm 1, line 8)
        self.block_edge_weight = self._project(
            budget, self.block_edge_weight, self.coeffs["eps"]
        )

        # For monitoring
        scalars = dict(
            prob_mass_after_update=pmass_update.sum().item(),
            prob_mass_after_update_max=pmass_update.max().item(),
            prob_mass_after_projection=self.block_edge_weight.sum().item(),
            prob_mass_after_projection_nonzero_weights=(
                self.block_edge_weight > self.coeffs["eps"]
            )
            .sum()
            .item(),
            prob_mass_after_projection_max=self.block_edge_weight.max().item(),
        )
        return scalars

    # @torch.no_grad()
    def _close(
        self,
        x: Tensor,
        labels: Tensor,
        budget: int,
        idx_attack: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Clean up and prepare return argument."""
        # Sample final discrete graph (Algorithm 1, line 16)
        edge_index, flipped_edges, edge_weight = self._sample_final_edges(
            x, labels, budget, idx_attack=idx_attack, **kwargs
        )

        return edge_index, flipped_edges, edge_weight

    def _forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, **kwargs
    ) -> Tensor:
        """Forward model."""
        return self.model(x, edge_index, edge_weight, **kwargs)

    def _forward_and_gradient(
        self, x: Tensor, labels: Tensor, idx_attack: Optional[Tensor] = None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Forward and update edge weights."""
        if not self.block_edge_weight.requires_grad:
            self.block_edge_weight.requires_grad = True

        # Retrieve sparse perturbed adjacency matrix `A \oplus p_{t-1}`
        # (Algorithm 1, line 6 / Algorithm 2, line 7)
        edge_index, edge_weight = self._get_modified_adj(
            self.edge_index,
            self.edge_weight,
            self.block_edge_index,
            self.block_edge_weight,
        )

        # Get prediction (Algorithm 1, line 6 / Algorithm 2, line 7)
        prediction = self._forward(x, edge_index, edge_weight, **kwargs)
        # Calculate loss combining all each node
        # (Algorithm 1, line 7 / Algorithm 2, line 8)
        loss = self.loss(prediction, labels, idx_attack)
        # Retrieve gradient towards the current block
        # (Algorithm 1, line 7 / Algorithm 2, line 8)
        gradient = torch.autograd.grad(loss, self.block_edge_weight)[0]

        return loss, gradient

    def _get_modified_adj(
        self,
        edge_index: Tensor,
        edge_weight: Tensor,
        block_edge_index: Tensor,
        block_edge_weight: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Merges adjacency matrix with current block (incl. weights)"""
        if self.is_undirected:
            block_edge_index, block_edge_weight = to_undirected(
                block_edge_index,
                block_edge_weight,
                num_nodes=self.num_nodes,
                reduce="mean",
            )

        modified_edge_index = torch.cat(
            (edge_index.to(self.device), block_edge_index), dim=-1
        )
        modified_edge_weight = torch.cat(
            (edge_weight.to(self.device), block_edge_weight)
        )

        modified_edge_index, modified_edge_weight = coalesce(
            modified_edge_index,
            modified_edge_weight,
            num_nodes=self.num_nodes,
            reduce="sum",
        )

        # Allow (soft) removal of edges
        is_edge_in_clean_adj = modified_edge_weight > 1
        modified_edge_weight[is_edge_in_clean_adj] = (
            2 - modified_edge_weight[is_edge_in_clean_adj]
        )

        # to_remove = modified_edge_weight <= 1e-12
        # modified_edge_weight -= modified_edge_weight * to_remove

        return modified_edge_index, modified_edge_weight

    def _filter_self_loops_in_block(self, with_weight: bool):
        is_not_sl = self.block_edge_index[0] != self.block_edge_index[1]
        self.current_block = self.current_block[is_not_sl]
        self.block_edge_index = self.block_edge_index[:, is_not_sl]
        if with_weight:
            self.block_edge_weight = self.block_edge_weight[is_not_sl]

    def _sample_random_block(self, budget: int = 0):
        for _ in range(self.coeffs["max_trials_sampling"]):
            num_possible_edges = self._num_possible_edges(
                self.num_nodes, self.is_undirected
            )
            self.current_block = torch.randint(
                num_possible_edges, (self.block_size,), device=self.device
            )
            self.current_block = torch.unique(self.current_block, sorted=True)
            if self.is_undirected:
                self.block_edge_index = self._linear_to_triu_idx(
                    self.num_nodes, self.current_block
                )
            else:
                self.block_edge_index = self._linear_to_full_idx(
                    self.num_nodes, self.current_block
                )
                self._filter_self_loops_in_block(with_weight=False)

            self.block_edge_weight = torch.full(
                self.current_block.shape, self.coeffs["eps"], device=self.device
            )

            # use weights poisoning edge_weight weights if edge is present there
            # for i, edge in enumerate(self.block_edge_index.T):
            #     if torch.any(torch.all(self.edge_index.T == edge, dim=1)):
            #         j = torch.where(torch.all(self.edge_index.T == edge,
            #                                 dim=-1))[0]
            #         self.block_edge_weight[i] = self.edge_weight[j]

            if self.current_block.size(0) >= budget:
                return

        raise RuntimeError(
            "Sampling random block was not successful. " "Please decrease `budget`."
        )

    def _resample_random_block(self, budget: int):
        # Keep at most half of the block (i.e. resample low weights)
        sorted_idx = torch.argsort(self.block_edge_weight)
        keep_above = (self.block_edge_weight <= self.coeffs["eps"]).sum().long()
        if keep_above < sorted_idx.size(0) // 2:
            keep_above = sorted_idx.size(0) // 2
        sorted_idx = sorted_idx[keep_above:]

        self.current_block = self.current_block[sorted_idx]

        # Sample until enough edges were drawn
        for _ in range(self.coeffs["max_trials_sampling"]):
            n_edges_resample = self.block_size - self.current_block.size(0)
            num_possible_edges = self._num_possible_edges(
                self.num_nodes, self.is_undirected
            )
            lin_index = torch.randint(
                num_possible_edges, (n_edges_resample,), device=self.device
            )

            current_block = torch.cat((self.current_block, lin_index))
            self.current_block, unique_idx = torch.unique(
                current_block, sorted=True, return_inverse=True
            )

            if self.is_undirected:
                self.block_edge_index = self._linear_to_triu_idx(
                    self.num_nodes, self.current_block
                )
            else:
                self.block_edge_index = self._linear_to_full_idx(
                    self.num_nodes, self.current_block
                )

            # Merge existing weights with new edge weights
            block_edge_weight_prev = self.block_edge_weight[sorted_idx]
            self.block_edge_weight = torch.full(
                self.current_block.shape, self.coeffs["eps"], device=self.device
            )
            self.block_edge_weight[unique_idx[: sorted_idx.size(0)]] = (
                block_edge_weight_prev
            )

            if not self.is_undirected:
                self._filter_self_loops_in_block(with_weight=True)

            if self.current_block.size(0) > budget:
                return
        raise RuntimeError(
            "Sampling random block was not successful." "Please decrease `budget`."
        )

    def _sample_final_edges(
        self,
        x: Tensor,
        labels: Tensor,
        budget: int,
        idx_attack: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        best_metric = float("-Inf")
        block_edge_weight = self.block_edge_weight
        block_edge_weight[block_edge_weight <= self.coeffs["eps"]] = 0

        for i in range(self.coeffs["max_final_samples"]):

            # block_edge_weight = torch.clamp(block_edge_weight, min=0, max=1)
            sampled_edges = torch.bernoulli(block_edge_weight).float()

            num_flips = (sampled_edges == 1).sum()
            if num_flips > budget:
                # Allowed budget is exceeded
                continue

            edge_index, edge_weight = self._get_modified_adj(
                self.edge_index, self.edge_weight, self.block_edge_index, sampled_edges
            )
            prediction = self._forward(x, edge_index, edge_weight, **kwargs)
            metric = self.metric(prediction, labels, idx_attack)

            # Save best sample
            if metric > best_metric:
                best_metric = metric
                self.block_edge_weight = sampled_edges.clone().cpu()

        # Recover best sample
        self.block_edge_weight = self.block_edge_weight.to(self.device)

        # flipped_edges = self.block_edge_index[:, self.block_edge_weight > 0]
        flipped_edges = self.block_edge_index[:, self.block_edge_weight == 1]

        edge_index, edge_weight = self._get_modified_adj(
            self.edge_index,
            self.edge_weight,
            self.block_edge_index,
            self.block_edge_weight,
        )

        edge_mask = edge_weight == 1.0
        edge_index = edge_index[:, edge_mask]
        edge_weight = edge_weight[edge_mask]

        return edge_index, flipped_edges, edge_weight

    def _update_edge_weights(
        self, budget: int, block_edge_weight: Tensor, epoch: int, gradient: Tensor
    ) -> Tensor:
        # The learning rate is refined heuristically, s.t. (1) it is
        # independent of the number of perturbations (assuming an undirected
        # adjacency matrix) and (2) to decay learning rate during fine-tuning
        # (i.e. fixed search space).
        lr = (
            budget
            / self.num_nodes
            * self.lr
            / np.sqrt(max(0, epoch - self.epochs_resampling) + 1)
        )
        return block_edge_weight + lr * gradient

    @staticmethod
    def _project(budget: int, values: Tensor, eps: float = 1e-7) -> Tensor:
        r"""Project :obj:`values`:
        :math:`budget \ge \sum \Pi_{[0, 1]}(\text{values})`."""
        if torch.clamp(values, 0, 1).sum() > budget:
            left = (values - 1).min()
            right = values.max()
            miu = PRBCDAttack._bisection(values, left, right, budget)
            values = values - miu
        return torch.clamp(values, min=eps, max=1 - eps)

    @staticmethod
    def _bisection(
        edge_weights: Tensor, a: float, b: float, n_pert: int, eps=1e-5, max_iter=1e3
    ) -> Tensor:
        """Bisection search for projection."""

        def shift(offset: float):
            return torch.clamp(edge_weights - offset, 0, 1).sum() - n_pert

        miu = a
        for _ in range(int(max_iter)):
            miu = (a + b) / 2
            # Check if middle point is root
            if shift(miu) == 0.0:
                break
            # Decide the side to repeat the steps
            if shift(miu) * shift(a) < 0:
                b = miu
            else:
                a = miu
            if (b - a) <= eps:
                break
        return miu

    @staticmethod
    def _num_possible_edges(n: int, is_undirected: bool) -> int:
        """Determine number of possible edges for graph."""
        if is_undirected:
            return n * (n - 1) // 2
        else:
            return int(n**2)  # We filter self-loops later

    @staticmethod
    def _linear_to_triu_idx(n: int, lin_idx: Tensor) -> Tensor:
        """Linear index to upper triangular matrix without diagonal. This is
        similar to
        https://stackoverflow.com/questions/242711/algorithm-for-index-numbers-of-triangular-matrix-coefficients/28116498#28116498
        with number nodes decremented and col index incremented by one."""
        nn = n * (n - 1)
        row_idx = (
            n
            - 2
            - torch.floor(
                torch.sqrt(-8 * lin_idx.double() + 4 * nn - 7) / 2.0 - 0.5
            ).long()
        )
        col_idx = (
            1
            + lin_idx
            + row_idx
            - nn // 2
            + torch.div((n - row_idx) * (n - row_idx - 1), 2, rounding_mode="floor")
        )
        return torch.stack((row_idx, col_idx))

    @staticmethod
    def _linear_to_full_idx(n: int, lin_idx: Tensor) -> Tensor:
        """Linear index to dense matrix including diagonal."""
        row_idx = torch.div(lin_idx, n, rounding_mode="floor")
        col_idx = lin_idx % n
        return torch.stack((row_idx, col_idx))

    @staticmethod
    def _margin_loss(
        score: Tensor,
        labels: Tensor,
        idx_mask: Optional[Tensor] = None,
        reduce: Optional[str] = None,
    ) -> Tensor:
        r"""Margin loss between true score and highest non-target score:

        .. math::
            m = - s_{y} + max_{y' \ne y} s_{y'}

        where :math:`m` is the margin :math:`s` the score and :math:`y` the
        labels.

        Args:
            score (Tensor): Some score (*e.g.*, logits) of shape
                :obj:`[n_elem, dim]`.
            labels (LongTensor): The labels of shape :obj:`[n_elem]`.
            idx_mask (Tensor, optional): To select subset of `score` and
                `labels` of shape :obj:`[n_select]`. Defaults to None.
            reduce (str, optional): if :obj:`mean` the result is aggregated.
                Otherwise, return element wise margin.

        :rtype: (Tensor)
        """
        if idx_mask is not None:
            score = score[idx_mask]
            labels = labels[idx_mask]

        linear_idx = torch.arange(score.size(0), device=score.device)
        true_score = score[linear_idx, labels]

        score = score.clone()
        score[linear_idx, labels] = float("-Inf")
        best_non_target_score = score.amax(dim=-1)

        margin_ = best_non_target_score - true_score
        if reduce is None:
            return margin_
        return margin_.mean()

    @staticmethod
    def _tanh_margin_loss(
        prediction: Tensor, labels: Tensor, idx_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate tanh margin loss, a node-classification loss that focuses
        on nodes next to decision boundary.

        Args:
            prediction (Tensor): Prediction of shape :obj:`[n_elem, dim]`.
            labels (LongTensor): The labels of shape :obj:`[n_elem]`.
            idx_mask (Tensor, optional): To select subset of `score` and
                `labels` of shape :obj:`[n_select]`. Defaults to None.

        :rtype: (Tensor)
        """
        log_prob = F.log_softmax(prediction, dim=-1)
        margin_ = GRBCDAttack._margin_loss(log_prob, labels, idx_mask)
        loss = torch.tanh(margin_).mean()
        return loss

    @staticmethod
    def _probability_margin_loss(
        prediction: Tensor, labels: Tensor, idx_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate probability margin loss, a node-classification loss that
        focuses  on nodes next to decision boundary. See `Are Defenses for
        Graph Neural Networks Robust?
        <https://www.cs.cit.tum.de/daml/are-gnn-defenses-robust>`_ for details.

        Args:
            prediction (Tensor): Prediction of shape :obj:`[n_elem, dim]`.
            labels (LongTensor): The labels of shape :obj:`[n_elem]`.
            idx_mask (Tensor, optional): To select subset of `score` and
                `labels` of shape :obj:`[n_select]`. Defaults to None.

        :rtype: (Tensor)
        """
        prob = F.softmax(prediction, dim=-1)
        margin_ = GRBCDAttack._margin_loss(prob, labels, idx_mask)
        return margin_.mean()

    @staticmethod
    def _masked_cross_entropy(
        log_prob: Tensor, labels: Tensor, idx_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Calculate masked cross entropy loss, a node-classification loss that
        focuses on nodes next to decision boundary.

        Args:
            log_prob (Tensor): Log probabilities of shape :obj:`[n_elem, dim]`.
            labels (LongTensor): The labels of shape :obj:`[n_elem]`.
            idx_mask (Tensor, optional): To select subset of `score` and
                `labels` of shape :obj:`[n_select]`. Defaults to None.

        :rtype: (Tensor)
        """
        if idx_mask is not None:
            log_prob = log_prob[idx_mask]
            labels = labels[idx_mask]

        is_correct = log_prob.argmax(-1) == labels
        if is_correct.any():
            log_prob = log_prob[is_correct]
            labels = labels[is_correct]

        return F.nll_loss(log_prob, labels)

    def _append_statistics(self, mapping: Dict[str, Any]):
        for key, value in mapping.items():
            self.attack_statistics[key].append(value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# poisoning with evasion
class PoisoningAttack(PRBCDAttack):
    def _forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, **kwargs
    ) -> Tensor:
        """Forward model."""
        self.model.reset_parameters()

        with torch.enable_grad():
            ped = copy.copy(data)
            ped.x, ped.edge_index, ped.edge_weight = x, edge_index, edge_weight
            train(self.model, ped, n_epochs, lr, weight_decay)

        self.model.eval()
        return self.model(x, edge_index, edge_weight)

    def _forward_and_gradient(
        self, x: Tensor, labels: Tensor, idx_attack: Optional[Tensor] = None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Forward and update edge weights."""
        if not self.block_edge_weight.requires_grad:
            self.block_edge_weight.requires_grad = True

        self.model.reset_parameters()

        self.model.train()
        opt = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        with higher.innerloop_ctx(self.model, opt) as (fmodel, diffopt):
            # apply poisoning perts

            edge_index, edge_weight = self._get_modified_adj(
                self.edge_index,
                self.edge_weight,
                self.block_edge_index,
                self.block_edge_weight,
            )

            # Normalize only once (only relevant if model normalizes adj)
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, num_nodes=data.num_nodes
            )
            if hasattr(fmodel, "norm"):
                edge_index, edge_weight_ = fmodel.norm(
                    edge_index,
                    edge_weight,
                    num_nodes=x.size(0),
                    add_self_loops=False,
                )

            # Train model on poisoned graph
            for _ in range(n_epochs):
                if hasattr(fmodel, "norm"):
                    pred = fmodel.forward(x, edge_index, edge_weight_, skip_norm=True)
                else:
                    pred = fmodel.forward(x, edge_index, edge_weight)
                loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])

                diffopt.step(loss)

            evasion_budget = self.budget

            grads = []
            for _ in range(args.evasion_reps):

                # edge_weight = torch.clamp(edge_weight, min=0, max=1)
                ev_edge_weight = (
                    torch.bernoulli(edge_weight) - edge_weight.detach() + edge_weight
                )

                self.evasion = EvasionAttack(
                    fmodel,
                    epochs=self.evasion_epochs,
                    block_size=BLOCK_SIZE,
                    metric=metric,
                    lr=self.evasion_lr,
                )
                self.evasion.edge_weight = ev_edge_weight

                (
                    ev_edge_index,
                    ev_perts,
                    ev_edge_weight,
                    rel_edge_index,
                    rel_edge_weight,
                ) = self.evasion.attack(
                    x, edge_index, labels, evasion_budget, idx_attack
                )

                self.ev_edge_weight = ev_edge_weight
                self.ev_loss += self.evasion.attack_statistics["loss"]

                # Get poisoned model predictions and compute loss
                pred = fmodel(x, ev_edge_index, ev_edge_weight)

                mu = self.block_edge_weight.mean()
                psum = self.block_edge_weight.sum()
                loss = (
                    self.loss(pred, labels, idx_attack)
                    + self.reg_coeff * (psum - mu) ** 2
                )

                gradient = torch.autograd.grad(loss, self.block_edge_weight)[0]
                grads.append(gradient)

                # evaluate model on evasion graph for internal stats
                with torch.no_grad():
                    # graph (w/ discrete adj matrix) after evasion
                    temp_data = copy.copy(data)
                    temp_data.edge_index = ev_edge_index

                    # get graph before final sampling
                    temp_cont_data = copy.copy(data)
                    rel_edge_index, rel_edge_weight = self._get_modified_adj(
                        self.edge_index,
                        self.edge_weight,
                        rel_edge_index,
                        rel_edge_weight,
                    )
                    temp_cont_data.edge_index = rel_edge_index
                    temp_cont_data.edge_weight = rel_edge_weight

                    model_acc = test(fmodel, temp_data)
                    cont_model_acc = test(fmodel, temp_cont_data)
                    self.model_accs.append(model_acc)
                    self.cont_model_accs.append(cont_model_acc)
                    self.disc_cont_gap.append(cont_model_acc - model_acc)

            gradient = torch.stack(grads).mean(dim=0)

        # Clip gradient for stability:
        clip_norm = 0.5
        grad_len_sq = gradient.square().sum()
        if grad_len_sq > clip_norm:
            gradient *= clip_norm / grad_len_sq.sqrt()

        self.model.eval()

        if self.return_ev:
            return loss, gradient, ev_edge_index, ev_edge_weight
        return loss, gradient


def get_transform(name):
    if name is None:
        return T.NormalizeFeatures()
    elif name == "jaccard":
        return T.Compose([T.NormalizeFeatures(), JaccardPurification()])
    elif name == "SVD":
        return T.Compose([T.NormalizeFeatures(), SVDPurification(K=50)])
    else:
        raise NotImplementedError


for _ in range(1):
    for (
        dataset_str,
        model_str,
        learning_rate,
        block_size,
        budget,
        reg_coeff,
        rep,
    ) in itertools.product(
        args.dataset,
        args.model,
        args.learning_rate,
        args.block_size,
        args.budget,
        args.reg_coeff,
        range(args.reps),
    ):

        vprint(f"Running {dataset_str} {model_str} {rep}")
        if dataset_str == "WikiCS":
            dataset = WikiCS(DATA_DIR, transform=T.NormalizeFeatures())
            data = dataset[0].to(device)
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
        elif dataset_str == "squirrel":
            dataset = WikipediaNetwork(
                DATA_DIR, name=dataset_str, transform=T.NormalizeFeatures()
            )
            data = dataset[0].to(device)
            data.train_mask = data.train_mask[:, 0]
            data.val_mask = data.val_mask[:, 0]
            data.test_mask = data.test_mask[:, 0]
        elif dataset_str == "arxiv":
            dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="../data")
            data = dataset[0].to(device)
            data.y = data.y.T[0]
            split_idx = dataset.get_idx_split()
            train_idx = split_idx["train"]
            valid_idx = split_idx["valid"]
            test_idx = split_idx["test"]
            data.train_mask = pyg.utils.index_to_mask(train_idx, size=data.num_nodes)
            data.val_mask = pyg.utils.index_to_mask(valid_idx, size=data.num_nodes)
            data.test_mask = pyg.utils.index_to_mask(test_idx, size=data.num_nodes)
        else:
            dataset = Planetoid(
                DATA_DIR, name=dataset_str, transform=get_transform(args.defense)
            )
            data = dataset[0].to(device)

        labeled_train, unlabeled_train, valset, testset = split_inductive(
            data.y.detach().cpu().numpy(),
            n_per_class=args.train_per_class,
            test_size=args.test_size,
        )

        if args.inductive:
            print("splitting inductively")
            train_nodes = torch.cat(
                (torch.from_numpy(labeled_train), torch.from_numpy(unlabeled_train))
            )
            trainval_nodes = torch.cat((train_nodes, torch.from_numpy(valset)))
            inductive_edge_index, _ = pyg.utils.subgraph(
                trainval_nodes, data.edge_index.detach().cpu()
            )
            inductive_edge_index = inductive_edge_index.to(device)
            # test_edges = [e for e in data.edge_index.T if e.tolist() not in inductive_edge_index.T.tolist()]
            test_edges = [
                e
                for e in data.edge_index.T
                if e.tolist()[0] in testset or e.tolist()[1] in testset
            ]
            test_edge_index = torch.stack(test_edges).T.to(device)
            # test_edge_index, _ = pyg.utils.subgraph(torch.from_numpy(testset), data.edge_index.detach().cpu())
            # test_edge_index = test_edge_index.to(device)
            assert (
                inductive_edge_index.shape[1] + test_edge_index.shape[1]
                == data.edge_index.shape[1]
            )
            data.inductive_mask = pyg.utils.mask.index_to_mask(
                trainval_nodes, data.num_nodes
            )

        data.train_mask = pyg.utils.mask.index_to_mask(
            torch.from_numpy(labeled_train), data.num_nodes
        )
        data.test_mask = pyg.utils.mask.index_to_mask(
            torch.from_numpy(testset), data.num_nodes
        )
        data.val_mask = pyg.utils.mask.index_to_mask(
            torch.from_numpy(valset), data.num_nodes
        )

        if args.wandb is not None:
            wandb.init(project=WANDB_PROJECT_NAME, reinit=True)
            config = wandb.config
            config.inductive = args.inductive
            config.block_size = block_size
            config.budget = budget
            config.poisoning_epochs = args.poisoning_epochs
            config.evasion_epochs = args.evasion_epochs
            config.inner_evasion_epochs = args.inner_evasion_epochs
            config.model = model_str
            config.evasion_reps = args.evasion_reps
            config.lr = learning_rate
            config.dataset = dataset_str
            config.defense = args.defense
            config.test_size = args.test_size
            config.train_per_class = args.train_per_class
            config.reg_coeff = reg_coeff

        results = dict()
        model = get_model(model_str, dataset).to(device)

        train(model, data, edge_index=inductive_edge_index if args.inductive else None)

        BUDGET = budget / 2
        BLOCK_SIZE = block_size
        global_budget = int(BUDGET * data.edge_index.size(1) / 2)

        clean_acc = test(model, data)
        results["clean_clean_acc"] = clean_acc

        poisoning_mask = data.val_mask if args.inductive else data.test_mask

        if "joint" in args.attack:
            model.reset_parameters()
            poisoning = PoisoningAttack(
                model,
                epochs=args.poisoning_epochs,
                block_size=BLOCK_SIZE,
                metric=metric,
                lr=learning_rate,
            )
            poisoning.return_ev = True
            poisoning.reg_coeff = reg_coeff
            poisoning.evasion_lr = learning_rate
            poisoning.evasion_epochs = args.inner_evasion_epochs
            poisoning.ev_loss = []
            poisoning.model_accs = []
            poisoning.cont_model_accs = []
            poisoning.disc_cont_gap = []

            print("budget:", global_budget)
            pois_edge_index, _, _, _ = poisoning.attack(
                data.x,
                data.edge_index,
                data.y,
                budget=global_budget,
                idx_attack=poisoning_mask,
            )
            pois_data = copy.copy(data)
            pois_data.edge_index = pois_edge_index

            model.reset_parameters()
            train(model, pois_data)

            if args.inductive:
                pois_data.edge_index = torch.cat(
                    [pois_data.edge_index, test_edge_index], dim=1
                )

            evasion = PRBCDAttack(
                model,
                epochs=args.poisoning_epochs,
                block_size=BLOCK_SIZE,
                metric=metric,
                lr=learning_rate,
            )
            ev_edge_index, _ = evasion.attack(
                data.x,
                pois_data.edge_index,
                data.y,
                budget=global_budget,
                idx_attack=data.test_mask,
            )
            ev_data = copy.copy(data)
            ev_data.edge_index = ev_edge_index

            # poisoned model on clean graph
            results["pois_clean_acc"] = test(model, data)

            # poisoned model on poisoned graph
            results["pois_pois_acc"] = test(model, pois_data)

            # poisoned model on poisoned + evasion graph
            results["pois_ev_acc"] = test(model, ev_data)

            if args.wandb is not None:
                wandb_log_list(
                    wandb, evasion.attack_statistics["loss"], "out_evasion_loss"
                )
                wandb_log_list(
                    wandb, poisoning.attack_statistics["loss"], "poisoning_loss"
                )
                wandb_log_list(wandb, poisoning.ev_loss, "in_evasion_loss")
                wandb_log_list(wandb, poisoning.cont_model_accs, "joint_cont_acc")
                wandb_log_list(wandb, poisoning.model_accs, "joint_disc_acc")
                wandb_log_list(wandb, poisoning.disc_cont_gap, "joint_gap")

        if "evasion" in args.attack:
            model.reset_parameters()
            train(model, data)

            ev_only_data = copy.copy(data)
            if args.inductive:
                ev_only_data.edge_index = torch.cat(
                    [ev_only_data.edge_index, test_edge_index], dim=1
                )

            evasion_o = PRBCDAttack(
                model,
                epochs=args.poisoning_epochs,
                block_size=BLOCK_SIZE,
                metric=metric,
                lr=learning_rate,
            )
            ev_only_edge_index, _ = evasion_o.attack(
                data.x,
                data.edge_index,
                data.y,
                budget=global_budget * 2,
                idx_attack=data.test_mask,
            )
            ev_only_data.edge_index = ev_only_edge_index

            results["clean_ev_acc"] = test(model, ev_only_data)

            if args.wandb is not None:
                wandb_log_list(
                    wandb, evasion_o.attack_statistics["loss"], "evasion_only_loss"
                )
        # ---------------

        if "sequential" in args.attack:
            model.reset_parameters()
            poisoning_normal = PoisoningPRBCDAttack(
                model,
                epochs=args.poisoning_epochs,
                block_size=BLOCK_SIZE,
                metric=metric,
                lr=learning_rate,
            )
            pois_normal_edge_index, _ = poisoning_normal.attack(
                data.x,
                data.edge_index,
                data.y,
                budget=global_budget,
                idx_attack=poisoning_mask,
            )
            pois_normal_data = copy.copy(data)
            pois_normal_data.edge_index = pois_normal_edge_index

            model.reset_parameters()
            train(model, pois_normal_data)

            # add test nodes after training
            if args.inductive:
                pois_normal_data.edge_index = torch.cat(
                    [pois_normal_data.edge_index, test_edge_index], dim=1
                )

            evasion_normal = PRBCDAttack(
                model,
                epochs=args.poisoning_epochs,
                block_size=BLOCK_SIZE,
                metric=metric,
                lr=learning_rate,
            )
            ev_normal_edge_index, _ = evasion_normal.attack(
                data.x,
                pois_normal_data.edge_index,
                data.y,
                budget=global_budget,
                idx_attack=data.test_mask,
            )
            ev_normal_data = copy.copy(data)
            ev_normal_data.edge_index = ev_normal_edge_index

            results["pois_normal_acc_clean"] = test(model, data)
            results["pois_normal_acc_pert"] = test(model, pois_normal_data)

            # poisoned (normal) model on evasion graph
            results["pois_normal_ev_acc"] = test(model, ev_normal_data)
        # ---------------

        if "poisoning" in args.attack:
            model.reset_parameters()
            pois_only = PoisoningPRBCDAttack(
                model,
                epochs=args.poisoning_epochs,
                block_size=BLOCK_SIZE,
                metric=metric,
                lr=learning_rate,
            )
            pois_normal_edge_index, _ = pois_only.attack(
                data.x,
                data.edge_index,
                data.y,
                budget=global_budget * 2,
                idx_attack=poisoning_mask,
            )
            pois_only_data = copy.copy(data)
            pois_only_data.edge_index = pois_normal_edge_index

            model.reset_parameters()
            train(model, pois_only_data)

            # add test nodes after training on poisoned graph
            if args.inductive:
                pois_only_data.edge_index = torch.cat(
                    [pois_only_data.edge_index, test_edge_index], dim=1
                )

            # poisoned (normal) model on evasion graph
            results["pois_only_acc_pert"] = test(model, pois_only_data)
            results["pois_only_acc_clean"] = test(model, data)

        if args.wandb is not None:
            wandb.log(results)
            wandb.finish()
        else:
            print(results)
