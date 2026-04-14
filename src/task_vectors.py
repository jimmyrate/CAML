import abc
from typing import OrderedDict, Union
from transformers import AutoModelForCausalLM
import torch

# from src.models.modeling import ImageEncoder
# from src.utils.variables_and_paths import MODELS

_Checkpoint = Union[str, dict, torch.nn.Module]


def symmetric_difference(A, B):
    """Returns the symmetric difference between two lists."""
    return list(set(A) ^ set(B))


class _TaskVector(abc.ABC):
    """
    Generic Task Vector class (Base class)
    """

    def __init__(self, model_name=None, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None):
        self.model_name = model_name

        # If task vector already provided
        if vector is not None:
            self.vector = vector
            return

        assert pretrained_checkpoint is not None and finetuned_checkpoint is not None

        with torch.no_grad():
            ptm = self._safe_load(pretrained_checkpoint)
            ftm = self._safe_load(finetuned_checkpoint)

            self.vector = {}
            for key in ptm:
                p = ptm[key]
                f = ftm[key]

                # ★ MODIFIED: skip non-tensor
                if not torch.is_tensor(p) or not torch.is_tensor(f):
                    continue

                # ★ MODIFIED: skip dtype not supporting subtraction
                if p.dtype in (torch.bool, torch.int32, torch.int64, torch.uint8):
                    continue
                if f.dtype in (torch.bool, torch.int32, torch.int64, torch.uint8):
                    continue

                # ★ MODIFIED: skip shape mismatch
                if p.shape != f.shape:
                    continue

                # ★ MODIFIED: compute task vector safely
                self.vector[key] = f - p

    # --------------------------
    # Safe loading methods
    # --------------------------

    def _safe_load(self, checkpoint):
        if isinstance(checkpoint, str):
            return self._load_checkpoint(checkpoint).state_dict()
        elif isinstance(checkpoint, dict):
            return checkpoint
        elif isinstance(checkpoint, torch.nn.Module):
            return checkpoint.state_dict()
        else:
            raise ValueError(f"Invalid checkpoint type: {type(checkpoint)}")

    @abc.abstractmethod
    def _load_checkpoint(self, path) -> torch.nn.Module:
        raise NotImplementedError

    # --------------------------
    # Vector operations
    # --------------------------

    def __add__(self, other):
        new_vec = {}
        for k in self.vector:
            if k in other.vector:
                new_vec[k] = self.vector[k] + other.vector[k]
        return self.__class__(vector=new_vec)

    def __neg__(self):
        return self.__class__(vector={k: -v for k, v in self.vector.items()})

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, coef):
        return self.__class__(vector={k: coef * v for k, v in self.vector.items()})

    def dot(self, other):
        s = 0.0
        for k in self.vector:
            if k in other.vector:
                s += torch.sum(self.vector[k] * other.vector[k])
        return s

    def norm(self):
        return torch.sqrt(self.dot(self))

    # --------------------------
    # Apply to model
    # --------------------------

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0, device="cuda"):
        """Apply task vector to a pretrained model"""
        model = self._load_checkpoint(pretrained_checkpoint)
        model = model.to(device)
        state = model.state_dict()
        for k in state:
            if k in self.vector:
                state[k] = state[k] + scaling_coef * self.vector[k]

        model.load_state_dict(state)
        return model


# ============================================================
# NON-LINEAR VERSION FOR HuggingFace (chemGPT / LLAMA / GPT2)
# ============================================================

class NonLinearTaskVector(_TaskVector):
    """Non-linear Task Vector for HF CausalLM models."""

    def _load_checkpoint(self, path):
        # ★ MODIFIED: nonlinear model loading
        return AutoModelForCausalLM.from_pretrained(path)