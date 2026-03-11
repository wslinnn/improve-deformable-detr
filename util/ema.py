"""
Simple EMA (Exponential Moving Average) module for model weights
Based on detrex implementation: https://github.com/IDEA-Research/detrex
"""
import torch
import itertools
from collections import OrderedDict
from typing import List


class EMAState(object):
    def __init__(self):
        self.state = {}

    @classmethod
    def FromModel(cls, model: torch.nn.Module, device: str = ""):
        ret = cls()
        ret.save_from(model, device)
        return ret

    def save_from(self, model: torch.nn.Module, device: str = ""):
        """Save model state from `model` to this object"""
        for name, val in self.get_model_state_iterator(model):
            val = val.detach().clone()
            self.state[name] = val.to(device) if device else val

    def apply_to(self, model: torch.nn.Module):
        """Apply state to `model` from this object"""
        with torch.no_grad():
            for name, val in self.get_model_state_iterator(model):
                assert (
                    name in self.state
                ), f"Name {name} not existed, available names {self.state.keys()}"
                val.copy_(self.state[name])

    def has_inited(self):
        return self.state

    def clear(self):
        self.state.clear()
        return self

    def get_model_state_iterator(self, model):
        param_iter = model.named_parameters()
        buffer_iter = model.named_buffers()
        return itertools.chain(param_iter, buffer_iter)

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict, strict: bool = True):
        self.state.clear()
        for x, y in state_dict.items():
            self.state[x] = y


class EMAUpdater(object):
    """Model Exponential Moving Average
    Keep a moving average of everything in the model state_dict (parameters and
    buffers).
    """

    def __init__(self, state: EMAState, decay: float = 0.999, device: str = ""):
        self.decay = decay
        self.device = device
        self.state = state

    def init_state(self, model):
        self.state.clear()
        self.state.save_from(model, self.device)

    def update(self, model):
        with torch.no_grad():
            ema_param_list = []
            param_list = []
            for name, val in self.state.get_model_state_iterator(model):
                ema_val = self.state.state[name]
                # Move to same device as model parameter
                if val.device != ema_val.device:
                    ema_val = ema_val.to(val.device)
                    self.state.state[name] = ema_val
                if val.dtype in [torch.float32, torch.float16]:
                    ema_param_list.append(ema_val)
                    param_list.append(val)
                else:
                    ema_val.copy_(ema_val * self.decay + val * (1.0 - self.decay))
            self._ema_avg(ema_param_list, param_list, self.decay)

    def _ema_avg(
        self,
        averaged_model_parameters: List[torch.Tensor],
        model_parameters: List[torch.Tensor],
        decay: float,
    ) -> None:
        """
        Function to perform exponential moving average:
        x_avg = alpha * x_avg + (1-alpha)* x_t
        """
        torch._foreach_mul_(averaged_model_parameters, decay)
        torch._foreach_add_(
            averaged_model_parameters, model_parameters, alpha=1 - decay
        )


class EMA:
    """
    Exponential Moving Average for model weights

    Args:
        model: PyTorch model
        decay: EMA decay rate (default: 0.999)
        device: device to store EMA weights
    """
    def __init__(self, model, decay=0.999, device=None):
        self.model = model
        self.decay = decay
        self.device = device

        # Use EMAState and EMAUpdater like detrex
        self.ema_state = EMAState()
        self.ema_updater = EMAUpdater(self.ema_state, decay=decay, device=device)
        self.ema_updater.init_state(model)

    def update(self):
        """Update shadow weights with exponential moving average"""
        self.ema_updater.update(self.model)

    def apply_shadow(self):
        """Replace model weights with shadow (EMA) weights"""
        # Save current model state before applying EMA
        self._backup = {}
        for name, val in self._get_model_state_iterator(self.model):
            self._backup[name] = val.detach().clone()
        # Move EMA state to same device as model before applying
        model_device = next(self.model.parameters()).device
        for name in self.ema_state.state:
            self.ema_state.state[name] = self.ema_state.state[name].to(model_device)
        # Apply EMA weights
        self.ema_state.apply_to(self.model)

    def restore(self):
        """Restore original model weights from backup"""
        if hasattr(self, '_backup') and self._backup:
            with torch.no_grad():
                for name, val in self._get_model_state_iterator(self.model):
                    # Ensure backup is on same device
                    if val.device != self._backup[name].device:
                        self._backup[name] = self._backup[name].to(val.device)
                    val.copy_(self._backup[name])
            self._backup = {}

    def _get_model_state_iterator(self, model):
        """Get iterator over all parameters and buffers"""
        param_iter = model.named_parameters()
        buffer_iter = model.named_buffers()
        return itertools.chain(param_iter, buffer_iter)

    @property
    def shadow(self):
        """Compatibility property for checkpoint saving"""
        return self.ema_state.state


def build_ema(model, args, device):
    """
    Build EMA for model

    Args:
        model: PyTorch model
        args: training arguments
        device: device

    Returns:
        EMA object if enabled, None otherwise
    """
    if not getattr(args, 'use_ema', False):
        return None

    decay = getattr(args, 'ema_decay', 0.999)
    ema = EMA(model, decay=decay, device=device)
    print(f"Using Model EMA with decay={decay}")
    return ema
