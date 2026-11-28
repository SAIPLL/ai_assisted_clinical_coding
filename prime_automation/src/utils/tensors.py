from typing import Any

import torch


def detach(x: torch.Tensor | Any) -> torch.Tensor | Any:
    """Detach a tensor from the computational graph"""
    if isinstance(x, torch.Tensor):
        return x.detach()
    return x
