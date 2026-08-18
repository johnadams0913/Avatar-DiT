import re
import sys
import torch
import os.path as osp
import itertools

from tqdm import tqdm
from omegaconf import OmegaConf
from safetensors.torch import save_file
from libs.convert_ckpt import convert_sd_ckpt
from model.util import rank_zero_print

VALID_FORMATS = [".pt", ".pth", ".ckpt", ".safetensors", ".bin"]


def get_format(filename):
    return osp.splitext(filename)[-1]

def fitting_weights(model, sd):
    n_params = len([name for name, _ in
                    itertools.chain(model.named_parameters(),
                                    model.named_buffers())])
    for name, param in tqdm(
            itertools.chain(model.named_parameters(),
                            model.named_buffers()),
            desc="Fitting old weights to new weights",
            total=n_params
    ):
        if not name in sd:
            continue
        old_shape = sd[name].shape
        new_shape = param.shape
        assert len(old_shape) == len(new_shape)
        if len(new_shape) > 2:
            # we only modify first two axes
            assert new_shape[2:] == old_shape[2:]
        # assumes first axis corresponds to output dim
        if not new_shape == old_shape:
            new_param = param.clone()
            old_param = sd[name]
            if len(new_shape) == 1:
                # Vectorized 1D case
                new_param = old_param[torch.arange(new_shape[0]) % old_shape[0]].clone()
            elif len(new_shape) >= 2:
                # Vectorized 2D case
                i_indices = torch.arange(new_shape[0])[:, None] % old_shape[0]
                j_indices = torch.arange(new_shape[1])[None, :] % old_shape[1]
                
                # Use advanced indexing to extract all values at once
                new_param = old_param[i_indices, j_indices].clone()
                
                # Count how many times each old column is used
                n_used_old = torch.bincount(
                    torch.arange(new_shape[1]) % old_shape[1], 
                    minlength=old_shape[1]
                )
                
                # Map to new shape
                n_used_new = n_used_old[torch.arange(new_shape[1]) % old_shape[1]]
                
                # Reshape for broadcasting
                n_used_new = n_used_new.reshape(1, new_shape[1])
                while len(n_used_new.shape) < len(new_shape):
                    n_used_new = n_used_new.unsqueeze(-1)
                
                # Normalize
                new_param = new_param / n_used_new

            sd[name] = new_param
    return sd


def load_config(path):
    # Load config file
    try:
        configs = OmegaConf.load(path)
    except:
        raise IOError("Failed in loading model configs, please check the training settings.")
    rank_zero_print(f"Loaded config from {path}")
    return configs


def load_weights(path):
    ext = get_format(path)
    assert ext in VALID_FORMATS, f"Invalid checkpoint format {ext}"
    if ext == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(path, device="cpu")
    else:
        sd = torch.load(path, map_location="cpu", mmap=True)
        if "state_dict" in sd.keys():
            sd = sd["state_dict"]
    return sd


def package_weights(filename, *args):
    sd = {}
    for path in args:
        print(path)
        sd.update(load_weights(path))
    save_file(sd, f"{filename}")


def delete_states(sd, delete_keys=list(), skip_keys=list()):
    keys = list(sd.keys())
    for k in keys:
        for ik in delete_keys:
            if len(skip_keys) > 0:
                for sk in skip_keys:
                    if re.search(ik, k) is not None and re.match(sk, k) is None:
                        rank_zero_print(f"Deleting key {k} from state_dict.")
                        del sd[k]
            else:
                if re.search(ik, k) is not None:
                    rank_zero_print(f"Deleting key {k} from state_dict.")
                    del sd[k]
    return sd


def filter_ema(sd):
    new_sd = {}
    for key in sd.keys():
        if key.find("cond_stage_model") > -1 or key.find("model_ema") > -1:
            continue
        if key.find("model.diffusion_model") > -1:
            new_sd[key] = sd[key.replace(".", "").replace("modeldiff", "model_ema.diff")].clone()
        new_sd[key] = sd[key].clone()
    return sd


def retrieve_ema(ckpt, filename=None):
    fmt = get_format(ckpt)
    sd = load_weights(ckpt)
    new_sd = filter_ema(sd)
    filename = osp.basename(ckpt.replace(fmt, ".safetensors")) if filename is None else filename
    save_file(new_sd, f"{filename}")


def postprocess_weights(ckpt_path, filename=None, exclude_frozen_params=True, ema=True):
    from libs.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
    sd = convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, exclude_frozen_parameters=exclude_frozen_params)

    if ema:
        new_sd = filter_ema(sd)
    else:
        new_sd = {}
        for k in sd:
            if k.find("cond_stage_model") > -1 or k.find("model_ema") > -1:
                continue
            new_sd[k] = sd[k].clone()

    filename = f"{osp.basename(ckpt_path)}.safetensors" if filename is None else filename
    save_file(new_sd, f"{filename}")


def post_with_ema(*args, **kwargs):
    postprocess_weights(ema=True, *args, **kwargs)

def post_without_ema(*args, **kwargs):
    postprocess_weights(ema=False, *args, **kwargs)


def filter_weights(ckpt, delete_keys, filename=None, skip_keys=None):
    skip_keys = [skip_keys] if skip_keys is not None else []
    fmt = get_format(ckpt)
    sd = load_weights(ckpt)
    sd = delete_states(sd, [delete_keys], skip_keys)
    filename = osp.basename(ckpt.replace(fmt, ".safetensors")) if filename is None else filename
    save_file(sd, f"{filename}")


def postprocess_distill_weights(ckpt_path, filename=None, model_index=1):
    """Extract generator weights from a distillation checkpoint.

    For SFT mode: model_1 is the generator (model_0 = frozen real score, model_2 = fake score).
    For LoRA mode (legacy): model_0/adapter_model.safetensors with fake_score keys filtered out.
    """
    if osp.isdir(ckpt_path):
        # SFT mode: generator is at model_{model_index}/model.safetensors
        sft_candidate = osp.join(ckpt_path, f"model_{model_index}", "model.safetensors")
        # LoRA mode (legacy): adapter weights at model_0/adapter_model.safetensors
        lora_candidate = osp.join(ckpt_path, "model_0", "adapter_model.safetensors")

        if osp.exists(sft_candidate):
            print(f"Loading SFT generator weights from {sft_candidate}")
            sd = load_weights(sft_candidate)
        elif osp.exists(lora_candidate):
            print(f"Loading LoRA adapter weights from {lora_candidate} (legacy mode)")
            sd = load_weights(lora_candidate)
            total_before = len(sd)
            sd = delete_states(sd, ["fake_score"])
            print(f"Filtered {total_before - len(sd)} fake_score keys, {len(sd)} keys remaining.")
        else:
            raise FileNotFoundError(
                f"No generator weights found in {ckpt_path}. "
                f"Checked: {sft_candidate}, {lora_candidate}"
            )
    else:
        sd = load_weights(ckpt_path)

    if filename is None:
        base = osp.basename(ckpt_path.rstrip("/"))
        filename = f"{base}_generator.safetensors"
    save_file(sd, filename)
    print(f"Saved {len(sd)} keys to {filename}")


def convert_ckpt(checkpoint, new_checkpoint):
    sd = load_weights(checkpoint)
    save_file(convert_sd_ckpt(sd), new_checkpoint)

def transfer_to_hf(config, ckpt_path, save_path):
    from model.util import instantiate_from_config
    model = instantiate_from_config(load_config(config).model)
    model.init_from_ckpt(ckpt_path)
    model.hf_save_model(save_path)

if __name__ == '__main__':
    functions = {
        "post": post_without_ema,
        "post-ema": post_with_ema,
        "merge": package_weights,
        "filter": filter_weights,
        "ema": retrieve_ema,
        "convert": convert_ckpt,
        "hfs": transfer_to_hf,
        "distill": postprocess_distill_weights,
    }
    args = sys.argv[1:]
    func = functions[args[0]]
    print(f"Applying util function {func.__name__}...")
    func(*args[1:])
    print(f"Process finished.")