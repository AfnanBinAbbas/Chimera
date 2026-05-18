from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from torch.utils.data import DataLoader

from src.dataset import RCAdatasets, get_eval_loader, get_loader


@dataclass
class MultiTaskExample:
    ad_label: int
    src_tokens: List[str]
    rca_mask: List[str]
    parse_tokens: Optional[List[str]] = None
    failure_type: Optional[int] = None
    impact_score: Optional[float] = None
    remediation_action: Optional[int] = None


def _split_fields(line: str) -> List[str]:
    return [part.strip() for part in line.strip().split(":") if part.strip() != ""]


def parse_multitask_line(line: str) -> Dict[str, object]:
    parts = _split_fields(line)
    if len(parts) < 3:
        raise ValueError(f"Expected at least 3 colon-separated fields, got {len(parts)}: {line!r}")

    ad_label = int(parts[0])
    src_tokens = parts[1].split()
    rca_mask = parts[2].split()

    parse_tokens = parts[3].split() if len(parts) > 3 else None
    failure_type = int(parts[4]) if len(parts) > 4 and parts[4] != "" else None
    impact_score = float(parts[5]) if len(parts) > 5 and parts[5] != "" else None
    remediation_action = int(parts[6]) if len(parts) > 6 and parts[6] != "" else None

    return {
        "ad_label": ad_label,
        "src_tokens": src_tokens,
        "rca_mask": rca_mask,
        "parse_tokens": parse_tokens,
        "failure_type": failure_type,
        "impact_score": impact_score,
        "remediation_action": remediation_action,
    }


def has_explicit_multitask_labels(root: str) -> bool:
    required = [
        os.path.join(root, "n_train_mt.txt"),
        os.path.join(root, "an_train_mt.txt"),
        os.path.join(root, "n_dev_mt.txt"),
        os.path.join(root, "an_dev_mt.txt"),
        os.path.join(root, "n_test_mt.txt"),
        os.path.join(root, "an_test_mt.txt"),
    ]
    return all(os.path.exists(path) for path in required)


def get_multitask_eval_loader(n_fp, batch_size: int = 4, shuffle: bool = False, num_workers: int = 0):
    return get_eval_loader(n_fp, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def get_weak_multitask_eval_loader(n_fp, batch_size: int = 4, shuffle: bool = False, num_workers: int = 0):
    # Keep batch structure compatible with Chimera-style forward() which expects
    # paired normal/anomaly-style batches.
    return get_loader(n_fp, n_fp, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
