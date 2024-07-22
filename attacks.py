import copy
import os.path as osp
from typing import Optional, Tuple, Union

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

import higher
import torch_geometric as pyg
from torch_geometric.contrib.nn.models.rbcd_attack import LOSS_TYPE
import torch_geometric.transforms as T
from torch_geometric.contrib.nn import GRBCDAttack
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import softmax, coalesce, to_undirected

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from tqdm import tqdm

from util import *
from models import *

LOSS_TYPE = Callable[[Tensor, Tensor, Optional[Tensor]], Tensor]


class PRBCDAttack(torch.nn.Module):
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
        return_ev: bool = False,
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
        self.return_ev = return_ev

        self.coeffs.update(kwargs)

    def attack(
        self,
        x: Tensor,
        edge_index: Tensor,
        labels: Tensor,
        budget: int,
        idx_attack: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
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
        # assert kwargs.get('edge_weight') is None
        edge_weight = (
            torch.ones(edge_index.size(1), device=self.device)
            if edge_weight is None
            else edge_weight
        )
        self.edge_index = edge_index.cpu().clone()
        self.edge_weight = edge_weight.cpu().clone()
        self.num_nodes = x.size(0)
        self.budget = budget

        # For collecting attack statistics
        self.attack_statistics = defaultdict(list)

        # Prepare attack and define `self.iterable` to iterate over
        step_sequence = self._prepare(budget)

        # Loop over the epochs (Algorithm 1, line 5)
        for step in tqdm(step_sequence, disable=not self.log, desc="Attack"):
            if self.return_ev:
                loss, gradient, self.ev_edge_index, self.ev_edge_weight = (
                    self._forward_and_gradient(
                        x, labels, idx_attack, return_ev=True, **kwargs
                    )
                )
            else:
                loss, gradient = self._forward_and_gradient(
                    x, labels, idx_attack, **kwargs
                )

            scalars = self._update(
                step, gradient, x, labels, budget, idx_attack, **kwargs
            )

            scalars["loss"] = loss.item()
            self._append_statistics(scalars)

        perturbed_edge_index, perturbed_edge_weight, flipped_edges = self._close(
            x, labels, budget, idx_attack, **kwargs
        )

        assert flipped_edges.size(1) <= budget, (
            f"# perturbed edges {flipped_edges.size(1)} " f"exceeds budget {budget}"
        )

        if self.return_ev:
            return (
                perturbed_edge_index,
                perturbed_edge_weight,
                self.ev_edge_index,
                self.ev_edge_weight,
            )
        return perturbed_edge_index, perturbed_edge_weight

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
        if not self.coeffs["with_early_stopping"]:
            return scalars

        # Calculate metric after the current epoch (overhead
        # for monitoring and early stopping)
        topk_block_edge_weight = torch.zeros_like(self.block_edge_weight)
        topk_block_edge_weight[torch.topk(self.block_edge_weight, budget).indices] = 1
        edge_index, edge_weight = self._get_modified_adj(
            self.edge_index,
            self.edge_weight,
            self.block_edge_index,
            topk_block_edge_weight,
        )
        prediction = self._forward(x, edge_index, edge_weight, **kwargs)
        metric = self.metric(prediction, labels, idx_attack)

        # Save best epoch for early stopping
        # (not explicitly covered by pseudo code)
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_block = self.current_block.cpu().clone()
            self.best_edge_index = self.block_edge_index.cpu().clone()
            self.best_pert_edge_weight = self.block_edge_weight.cpu().clone()

        # Resampling of search space (Algorithm 1, line 9-14)
        if epoch < self.epochs_resampling - 1:
            self._resample_random_block(budget)
        elif epoch == self.epochs_resampling - 1:
            # Retrieve best epoch if early stopping is active
            # (not explicitly covered by pseudo code)
            self.current_block = self.best_block.to(self.device)
            self.block_edge_index = self.best_edge_index.to(self.device)
            block_edge_weight = self.best_pert_edge_weight.clone()
            self.block_edge_weight = block_edge_weight.to(self.device)

        scalars["metric"] = metric.item()
        return scalars

    def _close(
        self,
        x: Tensor,
        labels: Tensor,
        budget: int,
        idx_attack: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """Clean up and prepare return argument."""
        # Retrieve best epoch if early stopping is active
        # (not explicitly covered by pseudo code)
        if self.coeffs["with_early_stopping"]:
            self.current_block = self.best_block.to(self.device)
            self.block_edge_index = self.best_edge_index.to(self.device)
            self.block_edge_weight = self.best_pert_edge_weight.to(self.device)

        # Sample final discrete graph (Algorithm 1, line 16)
        return self._sample_final_edges(
            x, labels, budget, idx_attack=idx_attack, **kwargs
        )

    def _forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, **kwargs
    ) -> Tensor:
        """Forward model."""
        return self.model(x, edge_index, edge_weight, **kwargs)

    def _forward_and_gradient(
        self, x: Tensor, labels: Tensor, idx_attack: Optional[Tensor] = None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Forward and update edge weights."""
        # self.block_edge_weight.requires_grad = True

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
                self.current_block.shape,
                self.coeffs["eps"],
                device=self.device,
                requires_grad=True,
            )
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
            if i == -1:
                # In first iteration employ top k heuristic instead of sampling
                sampled_edges = torch.zeros_like(block_edge_weight)
                sampled_edges[torch.topk(block_edge_weight, budget).indices] = 1
            else:
                # block_edge_weight = torch.clamp(block_edge_weight, 0, 1)
                block_edge_weight = torch.nan_to_num(block_edge_weight, nan=0)
                sampled_edges = torch.bernoulli(block_edge_weight).float()

            if sampled_edges.sum() > budget:
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
        flipped_edges = self.block_edge_index[:, self.block_edge_weight > 0]

        edge_index, edge_weight = self._get_modified_adj(
            self.edge_index,
            self.edge_weight,
            self.block_edge_index,
            self.block_edge_weight,
        )

        edge_mask = edge_weight > self.coeffs["eps"]
        edge_index = edge_index[:, edge_mask]
        edge_weight = edge_weight[edge_mask]

        return edge_index, edge_weight, flipped_edges

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


n_epochs = 50
lr = 0.04
weight_decay = 5e-4


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
                pred = fmodel.forward(x, edge_index, edge_weight, skip_norm=True)
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
