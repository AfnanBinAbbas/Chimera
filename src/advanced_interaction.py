from __future__ import annotations

import torch
import torch.nn as nn

from src.multitask_expansion import MultiTaskChimera


class AdvancedInteractionChimera(MultiTaskChimera):
    def __init__(self, cfg, embedding_dict, input_size=300, hidden_size=128):
        super().__init__(cfg, embedding_dict, input_size=input_size, hidden_size=hidden_size)
        self.enable_dynamic_interaction = True
        self.interaction_temperature = float(getattr(cfg, "interaction_temperature", 1.0))
        self.interaction_gate = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
        )

    def forward(self, batch):
        base_loss, pair = super().forward(batch)
        aux = pair[4] if len(pair) > 4 else {}
        losses = aux.get("losses", {}) if isinstance(aux, dict) else {}

        component_keys = [
            "base_loss",
            "parse_loss",
            "failure_loss",
            "impact_loss",
            "remediation_loss",
            "gan_loss",
        ]
        component_losses = [losses.get(key, torch.tensor(0.0, device=base_loss.device)) for key in component_keys]
        loss_vector = torch.stack([loss.reshape(1) if loss.dim() == 0 else loss.mean().reshape(1) for loss in component_losses]).view(1, -1)
        gate_logits = self.interaction_gate(torch.log1p(torch.clamp(loss_vector, min=0.0)) / max(self.interaction_temperature, 1e-6))
        interaction_weights = torch.softmax(gate_logits, dim=-1).view(-1)

        dynamic_loss = torch.sum(interaction_weights * torch.stack(component_losses))
        pair = pair + ({
            "interaction_weights": interaction_weights,
            "component_loss_names": component_keys,
        },)
        return dynamic_loss, pair
