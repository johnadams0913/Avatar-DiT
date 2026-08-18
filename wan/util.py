import os
import torch
import torch.nn as nn
import importlib

from inspect import isfunction
from functools import wraps
from torch.utils.checkpoint import checkpoint


def exists(x):
    return x is not None


def rank_zero_print(*args, **kwargs):
    """Print only on global rank 0. Works before torch.distributed is initialized
    because RANK env var is set by torchrun / accelerate launch at process start."""
    if int(os.environ.get("RANK", "0")) != 0:
        return
    print(*args, **kwargs)

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def expand_to_batch_size(x, bs):
    if isinstance(x, list):
        x = [cx.repeat(bs, *([1] * (len(cx.shape) - 1))) for cx in x]
    else:
        x = x.repeat(bs, *([1] * (len(x.shape) - 1)))
    return x

def torch_dfs(model: nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

def zero_drop(x, p, dim=0):
    return torch.bernoulli((1 - p) * append_dims(torch.ones(x.shape[dim], device=x.device), x.ndim)).to(x.dtype)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def checkpoint_wrapper(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Skip checkpointing when KV cache is active: in-place cache mutations
        # are not recomputation-safe (cache state changes between forward/recompute).
        use_ckpt = (not hasattr(self, 'checkpoint') or self.checkpoint) \
            and kwargs.get('kv_cache') is None
        if use_ckpt:
            def bound_func(*args, **kwargs):
                return func(self, *args, **kwargs)
            return checkpoint(bound_func, *args, use_reentrant=False, **kwargs)
        else:
            return func(self, *args, **kwargs)
    return wrapper

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def log_txt_as_img(wh, xc, size=16):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts