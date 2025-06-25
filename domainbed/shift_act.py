import copy
from copy import deepcopy
import torch
import torch.nn as nn
import torch.jit
import random
import torch.nn.functional as F
import numpy as np

class Shift_act(nn.Module):
    """Shift-act adapts a model by entropy minimization and prototypical contrastive loss during testing."""

    def __init__(self, model, optimizer, hparams, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.hparams = hparams
        self.device = hparams["device"]
        self.steps = steps
        self.iterations = 0
        self.initialize_state()

    def initialize_state(self):
        """Initialize model state and memory banks."""
        self.source_protos = self.model.classifier.weight.detach()
        self.source_labels = torch.arange(0, self.model.num_classes).to(self.device)
        self.reliable_features = self.model.classifier.weight.detach()
        self.reliable_labels = torch.arange(0, self.model.num_classes).to(self.device)
        self.dynamic_threshs = self.hparams["init_threshold"] * torch.ones(self.model.num_classes).to(self.device)
        self.reliable_counts = torch.ones(self.model.num_classes).to(self.device)
        self.true_counts = torch.zeros(self.model.num_classes).to(self.device)
        self.eval_counts = torch.zeros(self.model.num_classes).to(self.device)
        self.pred_confs = torch.zeros(self.model.num_classes).to(self.device)
        self.pred_counts = torch.zeros(self.model.num_classes).to(self.device)
        self.true_predicted = torch.zeros(self.model.num_classes).to(self.device)
        self.eval_confs = torch.zeros(self.model.num_classes).to(self.device)
        self.compute_eval_confs()

    def forward(self, x):
        """Forward pass and adaptation."""
        for step in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    def compute_eval_confs(self):
        """Compute evaluation confidences."""
        eval_logits = self.model.eval_logits
        eval_labels = self.model.eval_labels
        for c in range(self.model.num_classes):
            preds = torch.argmax(eval_logits, dim=1)
            mask = (eval_labels == c).int() * (preds == c).int()
            if mask.sum() == 0:
                self.eval_confs[c] = 0.0
                continue
            selected_logits = mask[:, None] * eval_logits
            selected_logits = selected_logits[(selected_logits != 0).any(dim=1)]
            probs = torch.softmax(selected_logits, dim=1)
            self.eval_confs[c] = probs[:, c].mean()

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward pass and model adaptation."""
        features, outputs, preds, max_vals = self.forward_pass(x)
        self.update_reliable_features_and_labels(features, preds, max_vals)
        self.update_dynamic_thresholds(preds, max_vals)
        ent_loss = self.compute_entropy_loss(outputs, preds, max_vals)
        pcl_loss = self.compute_pcl_loss(features)
        loss = ent_loss + self.hparams["trade_off"] * pcl_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.iterations += 1
        return outputs

    def forward_pass(self, x):
        """Perform a forward pass and extract key outputs."""
        features = self.model.featurizer(x)
        outputs = self.model.classifier(features)
        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)
        max_vals = torch.max(probs, dim=1).values
        return features, outputs, preds, max_vals

    def compute_entropy_loss(self, outputs, preds, max_vals):
        """Compute the entropy loss based on dynamic thresholds."""
        mask_reliable = (max_vals >= self.dynamic_threshs[preds]).int().detach()
        ent = softmax_entropy(outputs * mask_reliable[:, None])
        return ent.mean()

    def compute_pcl_loss(self, features):
        """Compute prototypical contrastive learning (PCL) loss."""
        mu_classes, std_classes = compute_gaussian_stats(
            self.reliable_features, self.reliable_labels, self.model.num_classes, self.source_protos
        )
        original_dists = get_distances(features, self.source_protos, dist_type="euclidean")
        candidate_dists, candidate_idx = torch.topk(original_dists, 3, dim=1, largest=False)
        diffs, mahalanobis_dists = [], []

        for i in range(3):
            diff = F.normalize((features - mu_classes[candidate_idx[:, i]]) / (std_classes[candidate_idx[:, i]] + 0.001), dim=1)
            diffs.append(diff)
            mahalanobis_dists.append(torch.sqrt((diff ** 2).sum(dim=1)))

        mahalanobis_dists = torch.stack(mahalanobis_dists, dim=1)
        min_dist = mahalanobis_dists.min(dim=1).values
        sims_min = torch.exp(-min_dist)
        sims_sum = torch.exp(-mahalanobis_dists).sum(dim=1)
        return -torch.log(sims_min / sims_sum).mean()

    def update_dynamic_thresholds(self, preds, max_vals):
        """Update dynamic thresholds based on predictions and confidences."""
        for c in range(self.model.num_classes):
            self.reliable_counts[c] = torch.sum(self.reliable_labels == c)
            self.eval_counts[c] = torch.sum(self.model.eval_labels == c)
            self.pred_counts[c] += torch.sum(preds == c)
            self.pred_confs[c] += torch.sum(max_vals * (preds == c).int())

        target_confs = self.pred_confs / torch.max(self.pred_counts, torch.tensor(1).to(self.device))
        confs_diff = target_confs / self.eval_confs
        self.dynamic_threshs += self.hparams["inc_rate"] * (1 - confs_diff)

    def update_reliable_features_and_labels(self, features, preds, max_vals):
        """Update reliable features and labels."""
        mask_reliable = (max_vals >= self.dynamic_threshs[preds]).int().detach()
        selected_features = mask_reliable[:, None] * features.detach()
        selected_features = selected_features[(selected_features != 0).any(dim=1)]
        selected_labels = mask_reliable * (preds + 1)
        non_zero_indices = torch.nonzero(selected_labels, as_tuple=True)
        selected_labels = selected_labels[non_zero_indices] - 1
        self.reliable_features = torch.cat((self.reliable_features, selected_features), dim=0)
        self.reliable_labels = torch.cat((self.reliable_labels, selected_labels), dim=0)
        
        
@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Compute entropy of softmax distribution."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def get_distances(X, Y, dist_type="euclidean"):
    """Compute distances between tensors X and Y."""
    if dist_type == "euclidean":
        return torch.cdist(X, Y)
    elif dist_type == "cosine":
        return 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    raise NotImplementedError(f"Distance type {dist_type} not supported.")

def compute_gaussian_stats(features, labels, num_classes, source_protos):
    """Compute Gaussian mean and variance for each class."""
    mus, stds = [], []
    for c in range(num_classes):
        mask = (labels == c).int()
        if mask.sum() < 10:
            mus.append(source_protos[c][None, :])
            stds.append(torch.zeros(features.size(1)).to(features.device)[None, :])
        else:
            selected_features = mask[:, None] * features
            selected_features = selected_features[(selected_features != 0).any(dim=1)]
            mus.append(selected_features.mean(dim=0, keepdim=True))
            stds.append(selected_features.var(dim=0, keepdim=True).sqrt())
    return torch.cat(mus, dim=0), torch.cat(stds, dim=0)

def configure_model(model, hparams):
    """Configure the model for adaptation."""
    model.train()
    for g in model.optimizer.param_groups:
        g["lr"] *= hparams["lr_coef"]
    model.network.requires_grad_(True)
    model.classifier.requires_grad_(False)
    return model
