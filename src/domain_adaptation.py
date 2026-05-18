from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function

from src.multitask_expansion import MultiTaskChimera


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, inputs, lambda_):
        ctx.lambda_ = lambda_
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(inputs, lambda_=1.0):
    return GradientReversalFunction.apply(inputs, lambda_)


class DomainAdaptiveChimera(MultiTaskChimera):
    def __init__(self, cfg, embedding_dict, input_size=300, hidden_size=128):
        super().__init__(cfg, embedding_dict, input_size=input_size, hidden_size=hidden_size)
        domain_hidden = getattr(cfg, "domain_hidden_size", 128)
        self.domain_loss_weight = 0.01  # Reduced weight to prevent task interference
        self.domain_temperature = float(getattr(cfg, "domain_temperature", 1.0))
        self.domain_discriminator = nn.Sequential(
            nn.Linear(hidden_size * 2, domain_hidden),
            nn.ReLU(),
            nn.Linear(domain_hidden, 2),
        )
        self.domain_loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        loss, pair = super().forward(batch)
        src, ad_label, rca_label = batch
        bag_src, _, _ = self.deal_batch(src, ad_label, rca_label)
        encoded = self.encode_streams(bag_src)

        domain_features = torch.cat(
            [self.pool_sequence(encoded["n_shared_out"]), self.pool_sequence(encoded["an_shared_out"])],
            dim=0,
        )
        domain_targets = torch.cat(
            [
                torch.zeros(encoded["bs"], dtype=torch.long, device=domain_features.device),
                torch.ones(encoded["bs"], dtype=torch.long, device=domain_features.device),
            ],
            dim=0,
        )

        # Dynamic domain weight scheduling (warm-up)
        current_epoch = getattr(self, "current_epoch", 0)
        scheduled_weight = self.domain_loss_weight * min(1.0, current_epoch / 50.0) if current_epoch > 10 else 0.0

        domain_logits = self.domain_discriminator(
            gradient_reversal(domain_features / max(self.domain_temperature, 1e-6), lambda_=scheduled_weight)
        )
        domain_loss = self.domain_loss_fn(domain_logits, domain_targets)
        total_loss = loss + scheduled_weight * domain_loss

        pair = pair + ({"domain_logits": domain_logits, "domain_target": domain_targets, "scheduled_domain_weight": scheduled_weight},)
        return total_loss, pair
