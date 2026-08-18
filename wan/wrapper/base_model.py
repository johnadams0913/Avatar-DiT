import torch
import time

from typing import Optional, Union
from einops import rearrange
from peft import LoraConfig, set_peft_model_state_dict
from peft.tuners.lora.layer import Linear as LoRALinearLayer

from ckpt_util import load_weights, delete_states, fitting_weights
from wan.util import instantiate_from_config, exists, rank_zero_print


def initialize_module(framework, model_name, config, last_time=None):
    if config is None:
        setattr(framework, model_name, None)
        rank_zero_print(f"Skip initializing {model_name} (config is None).")
        return last_time if last_time is not None else time.time()

    start_time = time.time()
    model = instantiate_from_config(config)
    setattr(framework, model_name, model)

    elapsed_time = time.time() - start_time
    if last_time is not None:
        total_time = time.time() - last_time
        rank_zero_print(f"Initialize {model_name}, cost time: {elapsed_time:.2f}s, total time: {total_time:.2f}s.")
    else:
        rank_zero_print(f"Initialize {model_name}, cost time: {elapsed_time:.2f}s.")

    return time.time()


def disable_train(self, mode=True):
    return self


def get_loraconfig(transformer, peft_params, **kwargs):
    target_modules = []
    include_keys = peft_params.pop("target_modules", None)
    exclude_keys = peft_params.pop("exclude_modules", None)

    if include_keys is None:
        include_keys = ["blocks"]
    if exclude_keys is None:
        exclude_keys = ["pose", "face", "camera", "motion"]
    for name, module in transformer.named_modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv3d, LoRALinearLayer)):
            continue
        if any(key in name for key in exclude_keys):
            continue
        if any(key in name for key in include_keys):
            target_modules.append(name)
    transformer_lora_config = LoraConfig(target_modules=target_modules, **peft_params, **kwargs)
    return transformer_lora_config


class BaseModel:
    vae: torch.nn.Module
    model: torch.nn.Module
    img_embedder: torch.nn.Module
    txt_embedder: torch.nn.Module
    control_encoder: Optional[torch.nn.Module]
    flame_adapter: Optional[torch.nn.Module]
    reference_net: Optional[torch.nn.Module]
    mv_adapter: Optional[torch.nn.Module]
    adapter_keys: list[str] = [
        "control_encoder",
        "flame_adapter",
        "reference_net",
        "mv_adapter"
    ]

    def __init__(
            self,
            device: str = "cuda",
            half_precision_dtype = "bfloat16",
            hf_load = False,
            dtype = torch.bfloat16,
            max_reft_len = 0,
            _skip_model_loading = False,
            *args,
            **kwargs
    ):
        self.device = device
        self.dtype = dtype
        self.hf_load = hf_load
        self.max_reft_len = max_reft_len
        self._skip_model_loading = _skip_model_loading

        rank_zero_print("Starting module initialization...")
        init_start_time = time.time()
        self.setup_models(*args, **kwargs)
        total_init_time = time.time() - init_start_time
        rank_zero_print(f"All modules initialized successfully. Total initialization time: {total_init_time:.2f}s.")

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.model.temporal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.model.temporal_downsample) if getattr(self, "vae", None) else 8
        self.half_precision_dtype = torch.bfloat16 if half_precision_dtype == "bfloat16" else torch.float16

    def setup_models(
            self,
            vae_config = None,
            transformer_config=None,
            img_embedder_config = None,
            txt_embedder_config = None,
            control_encoder_config = None,
            reference_net_config = None,
            flame_adapter_config = None,
            mv_adapter_config = None,
            lora_config = None,
            *args,
            **kwargs
    ):
        module_configs = [
            ("vae", vae_config, True),
            ("img_embedder", img_embedder_config, True),
            ("txt_embedder", txt_embedder_config, True),
            ("control_encoder", control_encoder_config, False),
            ("reference_net", reference_net_config, False),
            ("flame_adapter", flame_adapter_config, False),
            ("mv_adapter", mv_adapter_config, False)
        ]

        module_names = [name for name, _, _ in module_configs]
        if self.hf_load:
            module_start_time = time.time()
            import os
            from ..modules import WanAnimateModel
            ckpt_path = transformer_config.get("ckpt_path", None)
            params = transformer_config.get("params", {})

            if self._skip_model_loading:
                # FSDP with sync_module_states: create model without loading weights
                # Weights will be broadcast from rank 0 during accelerator.prepare()
                self.model = WanAnimateModel(**params)
                rank_zero_print(f"Skipping transformer weight loading (FSDP sync from rank 0)")
            elif exists(ckpt_path) and os.path.isfile(ckpt_path):
                self.model = WanAnimateModel(**params)
                rank_zero_print(f"Loading transformer weights from {ckpt_path}")
                weight_sd = load_weights(ckpt_path)
                missing, unexpected = self.model.load_state_dict(weight_sd, strict=False)
                rank_zero_print(f"Transformer weights loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
            elif exists(ckpt_path):
                self.model = WanAnimateModel.from_pretrained(
                    ckpt_path, low_cpu_mem_usage=False, **params
                )
            else:
                self.model = WanAnimateModel(**params)

            module_time = time.time() - module_start_time
            rank_zero_print(f"Initialize transformer, cost time: {module_time:.2f}s.")

        else:
            module_configs.append(("model", transformer_config, False))
        module_names.append("model")

        for module_name, config, frozen in module_configs:
            if exists(config):
                module_start_time = time.time()
                module = instantiate_from_config(config)
                if frozen:
                    module = module.eval().requires_grad_(False)
                    module.train = disable_train
                module_time = time.time() - module_start_time
                rank_zero_print(f"Initialize {module_name}, cost time: {module_time:.2f}s.")
            else:
                module = None
            setattr(self, module_name, module)

        self.merge_lora_weights = False
        if exists(lora_config):
            rank_zero_print("Start to hack lora layers")
            start_time = time.time()
            peft_lora_cfg = get_loraconfig(self.model, peft_params=lora_config.peft_params, **lora_config.get("kwargs", {}))
            # peft_lora_cfg = LoraConfig(**lora_config.peft_params)
            self.model.add_adapter(peft_lora_cfg)
            if hasattr(lora_config, "ckpt_path") and not self._skip_model_loading:
                rank_zero_print(f"Load lora weight from {lora_config.ckpt_path}")
                peft_sd = load_weights(lora_config.ckpt_path)
                set_peft_model_state_dict(self.model, peft_sd)
            self.merge_lora_weights = lora_config.get("merge", False)
            rank_zero_print(f"Lora modules initialized successfully. Merge and unload: {lora_config.get('merge', False)}. \
            Total initialization time: {time.time() - start_time:.2f}s.")

        else:
            self.lora = None

        self.model_dict = {name: getattr(self, name) for name in module_names}
        self.model_list: list[torch.nn.Module] = [
            module for name, module in self.model_dict.items() if exists(module)
        ]

    def get_video_length(self, frame_length, chunk_num):
        return (chunk_num - 1) * (frame_length - self.max_reft_len) + frame_length

    def train(self, mode = True):
        for m in self.model_list:
            m.train(mode)
        return self

    def eval(self):
        self.train(False)
        return self

    def cpu(self):
        for m in self.model_list:
            m.cpu()
        return self

    def cuda(self):
        for m in self.model_list:
            m.cuda()
        return self

    def switch_to_fp16(self):
        self.dtype = torch.bfloat16
        for m in self.model_list:
            m.to(self.half_precision_dtype)

    def switch_to_fp32(self):
        self.dtype = torch.float32
        for m in self.model_list:
            m.float()

    def init_from_ckpt(self, path, ignore_keys = list(), logging = False, make_it_fit = False):
        def filter_log(key, filter_keys):
            for fk in filter_keys:
                if key.find(fk) > -1:
                    return False
            return True

        sd = delete_states(load_weights(path), ignore_keys)
        if make_it_fit:
            sd = fitting_weights(self.model, sd)

        rank_zero_print(f"Start to loading model weights from {path}")
        with torch.no_grad():
            missing, unexpected = self.model.load_state_dict(sd, strict=False)

            for model_key in self.adapter_keys:
                if exists(self.model_dict[model_key]):
                    temp_sd = {}
                    for k in sd:
                        if model_key in k:
                            temp_sd[k.replace(f"{model_key}.", "")] = sd[k]
                    logs = self.model_dict[model_key].load_state_dict(temp_sd, strict=False)
                    missing += logs[0]
                    unexpected += logs[1]

        filtered_missing = []
        filtered_unexpect = []
        for k in missing:
            if filter_log(k, self.adapter_keys):
                filtered_missing.append(k)
        for k in unexpected:
            if filter_log(k, self.adapter_keys):
                filtered_unexpect.append(k)

        rank_zero_print(
            f"Restored from {path} with {len(filtered_missing)} filtered missing and "
            f"{len(filtered_unexpect)} filtered unexpected keys")
        if logging:
            if len(missing) > 0:
                rank_zero_print(f"Filtered missing Keys: {filtered_missing}")
            if len(unexpected) > 0:
                rank_zero_print(f"Filtered unexpected Keys: {filtered_unexpect}")

        if self.merge_lora_weights:
            self.model.fuse_lora()
            self.model.unload_lora()

    def get_i2v_mask(self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat([
            torch.repeat_interleave(msk[:, :1], repeats=4, dim=1), 
            msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)
        return msk

    def low_vram_shift(self, cuda_list: Union[str, list[str]]):
        if not isinstance(cuda_list, list):
            cuda_list = [cuda_list]
        cpu_list = self.model_dict.keys() - cuda_list
        for model in cpu_list:
            if exists(self.model_dict[model]):
                self.model_dict[model] = self.model_dict[model].cpu()
        torch.cuda.empty_cache()

        for model in cuda_list:
            if exists(self.model_dict[model]):
                self.model_dict[model] = self.model_dict[model].cuda()

    def hf_save_model(self, path, safe_serialization=True, max_shard_size="15GB"):
        self.model.save_pretrained(path, safe_serialization, max_shard_size)

    @torch.no_grad()
    def prepare_conditions(self, *args, **kwargs):
        raise NotImplementedError("Prepare conditions should be implemented in the subclass.")

    @torch.inference_mode()
    def generate_video(
            self,
            prompt,
            negative_prompt,
            control,
            reference,
            face,
            flame,
            video,
            camera,
            sampler,
            step: int,
            gs: float,
            height: int = 768,
            width: int = 512,
            low_vram: bool = True,
            frame_length: int = 81,
            chunk_num: int = 1,
            seed: int = None,
            video_length: int = None,
            deterministic: bool = True,
            motion_settings: dict = None,
            smpl = None,
            **kwargs,
    ):
        """
            User interface function.
        """
        if not low_vram:
            self.low_vram_shift([model for model in self.model_dict.keys()])

        cond, uncond = self.prepare_conditions(
            control,
            face = face,
            flame = flame,
            xc = reference,
            camera = camera,
            smpl = smpl,
            prompt = prompt,
            negative_prompt = negative_prompt,
            low_vram = low_vram,
            video = video,
            frame_length = frame_length,
            **motion_settings
        )

        if low_vram:
            self.low_vram_shift(["model"])

        video_length = (chunk_num - 1) * (frame_length - self.max_reft_len) + frame_length
        self.last_video_length = video_length

        reference_frames = 1
        if exists(reference) and isinstance(reference, torch.Tensor):
            reference_frames = reference.size(0)

        videos = self.sample(
            cfg_scale = gs,
            cond = cond,
            uncond = uncond,
            num_steps = step,
            seed = seed,
            shape = (height // 8, width // 8),
            offload_model = low_vram,
            sampler_type = sampler,
            frame_length = frame_length,
            chunk_num = chunk_num,
            reference_frames = reference_frames,
        )

        videos = torch.cat(videos, dim=1)
        rank_zero_print(f"videos.shape: {videos.shape}")
        videos = rearrange(videos, "c f h w -> f c h w")
        rank_zero_print(f"Video generation done. Total generated frames: {videos.shape[0]}")
        return videos.cpu()
