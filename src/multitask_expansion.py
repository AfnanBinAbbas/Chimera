from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import Chimera


class MultiTaskChimera(Chimera):
    def __init__(self, cfg, embedding_dict, input_size=300, hidden_size=128):
        super().__init__(cfg, embedding_dict, input_size=input_size, hidden_size=hidden_size)
        self.enable_gan = bool(getattr(cfg, "enable_gan", True))
        self.enable_aux_tasks = True
        self.num_failure_classes = int(getattr(cfg, "num_failure_classes", 4))
        self.num_remediation_classes = int(getattr(cfg, "num_remediation_classes", 6))
        self.aux_loss_weight = float(getattr(cfg, "aux_loss_weight", 0.35))
        self.gan_loss_weight = float(getattr(cfg, "gan_loss_weight", 0.15))
        self.parse_loss_weight = float(getattr(cfg, "parse_loss_weight", 0.20))
        self.failure_loss_weight = float(getattr(cfg, "failure_loss_weight", 0.25))
        self.impact_loss_weight = float(getattr(cfg, "impact_loss_weight", 0.10))
        self.remediation_loss_weight = float(getattr(cfg, "remediation_loss_weight", 0.15))
        self.gan_noise_dim = int(getattr(cfg, "gan_noise_dim", 64))
        gan_hidden = int(getattr(cfg, "gan_hidden_size", 128))

        self.parse_head = nn.Sequential(
            nn.Linear(256, gan_hidden),
            nn.ReLU(),
            nn.Linear(gan_hidden, 300),
        )
        self.failure_head = nn.Sequential(
            nn.Linear(256, gan_hidden),
            nn.ReLU(),
            nn.Linear(gan_hidden, self.num_failure_classes),
        )
        self.impact_head = nn.Sequential(
            nn.Linear(256, gan_hidden),
            nn.ReLU(),
            nn.Linear(gan_hidden, 1),
            nn.Sigmoid(),
        )
        self.remediation_head = nn.Sequential(
            nn.Linear(256, gan_hidden),
            nn.ReLU(),
            nn.Linear(gan_hidden, self.num_remediation_classes),
        )
        self.gan_generator = nn.Sequential(
            nn.Linear(self.gan_noise_dim, gan_hidden),
            nn.ReLU(),
            nn.Linear(gan_hidden, 256),
        )
        self.gan_discriminator = nn.Sequential(
            nn.Linear(256, gan_hidden),
            nn.ReLU(),
            nn.Linear(gan_hidden, 1),
        )

        self.parse_loss_fn = nn.MSELoss()
        self.failure_loss_fn = nn.CrossEntropyLoss()
        self.impact_loss_fn = nn.MSELoss()
        self.remediation_loss_fn = nn.CrossEntropyLoss()
        self.gan_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        base = self.base_forward(batch)
        full_features = torch.cat([base["n_z"], base["an_z"]], dim=0)
        weak_aux = base["weak_aux"]

        parse_pred = self.parse_head(full_features)
        failure_logits = self.failure_head(full_features)
        impact_pred = self.impact_head(full_features)
        remediation_logits = self.remediation_head(full_features)

        parse_loss = self.parse_loss_fn(parse_pred, weak_aux["parse_target"])
        failure_loss = self.failure_loss_fn(failure_logits, weak_aux["failure_target"])
        impact_loss = self.impact_loss_fn(impact_pred, weak_aux["impact_target"])
        remediation_loss = self.remediation_loss_fn(remediation_logits, weak_aux["remediation_target"])

        aux_loss = (
            self.parse_loss_weight * parse_loss
            + self.failure_loss_weight * failure_loss
            + self.impact_loss_weight * impact_loss
            + self.remediation_loss_weight * remediation_loss
        )

        gan_loss = torch.tensor(0.0, device=full_features.device)
        if self.training and self.enable_gan:
            noise = torch.randn(base["an_z"].size(0), self.gan_noise_dim, device=full_features.device)
            fake_an_z = self.gan_generator(noise)

            real_logits = self.gan_discriminator(base["an_z"])
            fake_logits = self.gan_discriminator(fake_an_z.detach())
            real_targets = torch.ones_like(real_logits)
            fake_targets = torch.zeros_like(fake_logits)
            disc_loss = self.gan_loss_fn(real_logits, real_targets) + self.gan_loss_fn(fake_logits, fake_targets)
            gen_loss = self.gan_loss_fn(self.gan_discriminator(fake_an_z), real_targets)

            augmented_logits = self.classifier(fake_an_z)
            augmented_targets = torch.ones(fake_an_z.size(0), dtype=torch.long, device=full_features.device)
            augmented_classification_loss = self.ce_loss(augmented_logits, augmented_targets)

            gan_loss = 0.5 * disc_loss + gen_loss + augmented_classification_loss

        total_loss = base["loss"] + self.aux_loss_weight * aux_loss + self.gan_loss_weight * gan_loss
        pair = base["pair"] + ({
            "parse_pred": parse_pred,
            "failure_logits": failure_logits,
            "impact_pred": impact_pred,
            "remediation_logits": remediation_logits,
            "parse_target": weak_aux["parse_target"],
            "failure_target": weak_aux["failure_target"],
            "impact_target": weak_aux["impact_target"],
            "remediation_target": weak_aux["remediation_target"],
            "losses": {
                "base_loss": base["loss"],
                "parse_loss": parse_loss,
                "failure_loss": failure_loss,
                "impact_loss": impact_loss,
                "remediation_loss": remediation_loss,
                "gan_loss": gan_loss,
                "aux_loss": aux_loss,
            },
        },)
        return total_loss, pair
