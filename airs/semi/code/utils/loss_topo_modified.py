import warnings
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# BCE loss
class BCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss(weight, size_average)

    def forward(self, pred, target):
        sizep = pred.size(0)
        sizet = target.size(0)
        pred_flat = pred.view(sizep, -1)
        target_flat = target.view(sizet, -1)

        loss = self.bceloss(pred_flat, target_flat)
        return loss


# Dice loss
class DiceLoss(nn.Module):
    def __init__(self, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = target.size(0)

        pred_flat = pred.view(size, -1)
        target_flat = target.view(size, -1)

        intersection = pred_flat * target_flat
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size
        return dice_loss


# BCE + Dice loss
class BceDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss(weight, size_average)
        self.dice = DiceLoss(size_average)

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        loss = diceloss + bceloss
        return loss


class GlobalCosineLoss(nn.Module):
    def __init__(self, weight, stop_grad=True):
        super(GlobalCosineLoss, self).__init__()
        self.cos_loss = nn.CosineSimilarity()
        self.weight = weight
        self.stop_grad = stop_grad

    def forward(self, a, b):
        loss = 0
        for item in range(len(a)):
            if self.stop_grad:
                loss += torch.mean(
                    1 - self.cos_loss(a[item].view(a[item].shape[0], -1).detach(),
                                      b[item].view(b[item].shape[0], -1))
                ) * self.weight[item]
            else:
                loss += torch.mean(
                    1 - self.cos_loss(a[item].view(a[item].shape[0], -1),
                                      b[item].view(b[item].shape[0], -1))
                ) * self.weight[item]
        return loss


import os
import sys
import warnings
import importlib
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


class BettiMatchingLoss(nn.Module):
    """
    Strict topo loss wrapper for the pybind Betti Matching backend.

    Notes:
    1) prediction should be probability map in [0, 1]
    2) target should be binary / near-binary
    3) matching backend itself is outside autograd
    4) gradients flow through selected prediction pixels after coordinates are returned
    5) input shape:
         [B, 1, H, W]
       (also works for [B, 1, D, H, W] if backend returns 3D coords)
    """

    def __init__(
        self,
        homology_dim: int = 0,
        matched_weight: float = 1.0,
        unmatched_weight: float = 1.0,
        include_target_unmatched: bool = False,
        backend_import: str = "build.betti_matching",
        backend_root: str = "/root/example/fyp/airs/Betti-Matching-3D-master",
        warn_once: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.homology_dim = homology_dim
        self.matched_weight = matched_weight
        self.unmatched_weight = unmatched_weight
        self.include_target_unmatched = include_target_unmatched
        self.backend_import = backend_import
        self.backend_root = backend_root
        self.warn_once = warn_once
        self.eps = eps

        self._backend = None
        self._backend_error = None
        self._warned_backend_missing = False
        self._init_backend()

    def _init_backend(self):
        try:
            # 关键：先把 BettiMatching 项目根目录加入 sys.path
            # 因为我们要 import 的是 build.betti_matching
            # 所以 sys.path 里应该放：
            #   /root/example/fyp/airs/Betti-Matching-3D-master
            if self.backend_root and os.path.isdir(self.backend_root):
                if self.backend_root not in sys.path:
                    sys.path.insert(0, self.backend_root)

            module = importlib.import_module(self.backend_import)
            self._backend = module
            self._backend_error = None

        except Exception as e:
            self._backend = None
            self._backend_error = e

    def backend_available(self) -> bool:
        return self._backend is not None and hasattr(self._backend, "compute_matching")

    def _warn_backend_missing(self):
        if self.warn_once and self._warned_backend_missing:
            return

        extra = ""
        if self._backend_error is not None:
            extra = f" Import error: {repr(self._backend_error)}"

        warnings.warn(
            f"Betti matching backend '{self.backend_import}' is not available. "
            f"Topo loss will be 0. Please compile/install the backend first. "
            f"backend_root='{self.backend_root}'.{extra}",
            RuntimeWarning,
        )
        self._warned_backend_missing = True

    @staticmethod
    def _get_attr(result: Any, name: str, default=None):
        if isinstance(result, dict):
            return result.get(name, default)
        return getattr(result, name, default)

    @staticmethod
    def _is_scalar_like(x: Any) -> bool:
        return isinstance(x, (int, float, np.integer, np.floating))

    def _select_dim_coords(self, coords_all: Any) -> List[Tuple[int, ...]]:
        if coords_all is None:
            return []

        if hasattr(coords_all, "tolist"):
            coords_all = coords_all.tolist()

        coords = coords_all

        # 兼容 backend 可能返回 [dim0_coords, dim1_coords, ...]
        if isinstance(coords_all, (list, tuple)) and len(coords_all) > 0:
            first = coords_all[0]
            if isinstance(first, (list, tuple)):
                if len(first) == 0:
                    coords = coords_all[self.homology_dim] if self.homology_dim < len(coords_all) else []
                else:
                    first0 = first[0]
                    if isinstance(first0, (list, tuple, np.ndarray)):
                        coords = coords_all[self.homology_dim] if self.homology_dim < len(coords_all) else []

        if hasattr(coords, "tolist"):
            coords = coords.tolist()

        parsed = []
        for c in coords:
            if hasattr(c, "tolist"):
                c = c.tolist()

            if isinstance(c, (list, tuple)):
                if len(c) == 0:
                    continue

                if self._is_scalar_like(c[0]):
                    parsed.append(tuple(int(v) for v in c))
                else:
                    # 有的 backend 可能多包一层
                    if len(c) > 0 and isinstance(c[0], (list, tuple, np.ndarray)):
                        cc = c[0].tolist() if hasattr(c[0], "tolist") else c[0]
                        parsed.append(tuple(int(v) for v in cc))

        return parsed

    @staticmethod
    def _index_tensor(x: torch.Tensor, coord: Sequence[int]) -> torch.Tensor:
        idx = tuple(int(v) for v in coord)
        return x[idx]

    def _sample_loss(self, pred_map: torch.Tensor, target_map: torch.Tensor) -> torch.Tensor:
        pred_np = pred_map.detach().cpu().numpy().astype(np.float64)
        target_np = target_map.detach().cpu().numpy().astype(np.float64)

        result = self._backend.compute_matching(
            pred_np,
            target_np,
            include_input2_unmatched_pairs=self.include_target_unmatched,
        )

        pb = self._select_dim_coords(self._get_attr(result, "input1_matched_birth_coordinates", []))
        pd = self._select_dim_coords(self._get_attr(result, "input1_matched_death_coordinates", []))
        tb = self._select_dim_coords(self._get_attr(result, "input2_matched_birth_coordinates", []))
        td = self._select_dim_coords(self._get_attr(result, "input2_matched_death_coordinates", []))

        upb = self._select_dim_coords(self._get_attr(result, "input1_unmatched_birth_coordinates", []))
        upd = self._select_dim_coords(self._get_attr(result, "input1_unmatched_death_coordinates", []))

        matched_terms = []
        num_matched = min(len(pb), len(pd), len(tb), len(td))
        for i in range(num_matched):
            pred_birth = self._index_tensor(pred_map, pb[i])
            pred_death = self._index_tensor(pred_map, pd[i])
            tgt_birth = self._index_tensor(target_map, tb[i])
            tgt_death = self._index_tensor(target_map, td[i])

            matched_terms.append(
                (pred_birth - tgt_birth) ** 2 +
                (pred_death - tgt_death) ** 2
            )

        unmatched_terms = []
        num_unmatched = min(len(upb), len(upd))
        for i in range(num_unmatched):
            pred_birth = self._index_tensor(pred_map, upb[i])
            pred_death = self._index_tensor(pred_map, upd[i])

            unmatched_terms.append((pred_birth - pred_death) ** 2)

        if len(matched_terms) > 0:
            matched_loss = torch.stack(matched_terms).mean()
        else:
            matched_loss = pred_map.new_tensor(0.0)

        if len(unmatched_terms) > 0:
            unmatched_loss = torch.stack(unmatched_terms).mean()
        else:
            unmatched_loss = pred_map.new_tensor(0.0)

        total = self.matched_weight * matched_loss + self.unmatched_weight * unmatched_loss
        return total

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred is None or target is None:
            raise ValueError("BettiMatchingLoss received None input.")
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch in BettiMatchingLoss: pred {pred.shape}, target {target.shape}")
        if pred.dim() < 4:
            raise ValueError("BettiMatchingLoss expects [B, C, ...] tensors.")
        if pred.size(1) != 1 or target.size(1) != 1:
            raise ValueError("BettiMatchingLoss currently expects single-channel binary maps.")

        if not self.backend_available():
            self._warn_backend_missing()
            return pred.new_tensor(0.0)

        # target 作为拓扑参照，统一转二值
        target = (target > 0.5).float()

        batch_losses = []
        for b in range(pred.size(0)):
            pred_map = pred[b, 0]
            target_map = target[b, 0]
            batch_losses.append(self._sample_loss(pred_map, target_map))

        if len(batch_losses) == 0:
            return pred.new_tensor(0.0)

        return torch.stack(batch_losses).mean()
