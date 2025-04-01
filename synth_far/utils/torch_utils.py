import torch
import numpy as np
import random
from typing import Dict, Any, List, Union
import pydoc
from torch import nn

from .logging import get_logger

logger = get_logger()

def set_determenistic(seed: int=None) -> None:
    if seed is None:
        logger.info("Skipping seed setting. Training will not be deterministic.")
        return
    logger.info(f"Setting seed {seed}")
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def dict_to_object(d: Dict[str, Any], parent=None, **default_kwargs) -> object:
    # add support for nested dicts to be converted to objects (at least for recursion depth of 2-3)
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skipcq PTC-W0034

    return pydoc.locate(object_type)(**kwargs)


def set_requires_grad(nets: Union[List[nn.Module], nn.Module], requires_grad: bool = False) -> None:
    """Set requies_grad for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


class EndlessIterator:
    def __init__(self, iterable):
        assert len(iterable) > 0
        self.iterable = iterable
        self.current_iterator = None

    def __next__(self):
        if self.current_iterator is None:
            self._update_iter()
        try:
            data = next(self.current_iterator)
            return data
        except StopIteration:
            self.current_iterator = None
            return next(self)

    def _update_iter(self):
        self.current_iterator = iter(self.iterable)
