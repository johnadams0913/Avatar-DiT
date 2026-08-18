import os
import time
import gc

import wandb
import torch
import torchvision
import numpy as np
import PIL.Image as Image

from glob import glob
from tqdm import tqdm
from wan.util import default
from accelerate import Accelerator
from safetensors.torch import save_file as save_safetensors

MAXM_SAMPLE_SIZE = 8
ckpt_fmt = "safetensors"
log_filename = "logs.txt"


def _get_fsdp_full_state_dict(accelerator, fsdp_model):
    """Get the full (gathered) state dict from an FSDP-wrapped model."""
    from accelerate.utils import DistributedType
    if accelerator.distributed_type == DistributedType.FSDP:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        full_sd_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, full_sd_config):
            return fsdp_model.state_dict()
    else:
        return accelerator.unwrap_model(fsdp_model).state_dict()


def custom_save_state(accelerator: Accelerator, output_dir: str):
    """
    Custom save_state that handles mismatched optimizer/model counts.
    Supports both LoRA (legacy) and full SFT training modes.
    Frozen models (no trainable params) are skipped to save disk space.
    Uses FSDP FullStateDictConfig to properly gather sharded parameters.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model(s)
    for i, model in enumerate(accelerator._models):
        model_path = os.path.join(output_dir, f"model_{i}")
        os.makedirs(model_path, exist_ok=True)

        unwrapped_model = accelerator.unwrap_model(model)

        # Skip fully frozen models
        if not any(p.requires_grad for p in unwrapped_model.parameters()):
            continue

        # Use FSDP full state dict gathering (handles sharded params correctly)
        state_dict = _get_fsdp_full_state_dict(accelerator, model)

        # Only rank 0 saves (full state dict is gathered there)
        if not accelerator.is_main_process:
            continue

        # Check for LoRA weights (backward compatibility)
        lora_state = {k: v for k, v in state_dict.items() if "lora_" in k}
        if lora_state:
            default_state = {k: v for k, v in lora_state.items() if "fake_score" not in k}
            aux_state = {k: v for k, v in lora_state.items() if "fake_score" in k}
            save_safetensors(default_state, os.path.join(model_path, "adapter_model.safetensors"))
            if aux_state:
                save_safetensors(aux_state, os.path.join(model_path, "adapter_model_aux.safetensors"))
        else:
            # SFT mode: save full model state
            save_safetensors(state_dict, os.path.join(model_path, "model.safetensors"))

    # Save optimizer(s) - each optimizer saved separately
    for i, optimizer in enumerate(accelerator._optimizers):
        optimizer_path = os.path.join(output_dir, f"optimizer_{i}.bin")
        torch.save(optimizer.state_dict(), optimizer_path)

    # Save scheduler(s) if any
    for i, scheduler in enumerate(accelerator._schedulers):
        scheduler_path = os.path.join(output_dir, f"scheduler_{i}.bin")
        torch.save(scheduler.state_dict(), scheduler_path)

    # Save random states
    random_states = {
        "random_state": torch.get_rng_state(),
        "numpy_random_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        random_states["cuda_random_state"] = torch.cuda.get_rng_state_all()
    torch.save(random_states, os.path.join(output_dir, "random_states.bin"))


def _load_safetensors_robust(path):
    """Load safetensors file, with fallback for files with trailing extra data."""
    from safetensors.torch import load_file as load_safetensors
    try:
        return load_safetensors(path)
    except Exception:
        import struct, json, numpy as np
        tqdm.write(f"Standard safetensors load failed for {path}, using manual parser...")
        sd = {}
        with open(path, 'rb') as f:
            hdr_len = struct.unpack('<Q', f.read(8))[0]
            metadata = json.loads(f.read(hdr_len).decode('utf-8'))
            data_start = 8 + hdr_len
            for key, info in metadata.items():
                if key == '__metadata__':
                    continue
                start, end = info['data_offsets']
                if end - start == 0:
                    continue  # skip empty tensors
                f.seek(data_start + start)
                raw = f.read(end - start)
                dtype_str = info['dtype']
                shape = info['shape']
                if dtype_str == 'BF16':
                    arr = np.frombuffer(raw, dtype=np.uint16).copy()
                    sd[key] = torch.from_numpy(arr).view(torch.bfloat16).reshape(shape)
                elif dtype_str == 'F32':
                    arr = np.frombuffer(raw, dtype=np.float32).copy()
                    sd[key] = torch.from_numpy(arr).reshape(shape)
                elif dtype_str == 'F16':
                    arr = np.frombuffer(raw, dtype=np.float16).copy()
                    sd[key] = torch.from_numpy(arr).view(torch.float16).reshape(shape)
        tqdm.write(f"  Manually loaded {len(sd)} tensors from {path}")
        return sd


def custom_load_state(accelerator: Accelerator, input_dir: str):
    """
    Custom load_state that handles mismatched optimizer/model counts.
    Supports both LoRA (legacy) and full SFT checkpoint formats.
    Frozen models (no saved weights) are skipped.
    """
    # Load model(s)
    for i, model in enumerate(accelerator._models):
        model_path = os.path.join(input_dir, f"model_{i}")
        if not os.path.exists(model_path):
            continue  # Frozen model, skipped during save

        unwrapped_model = accelerator.unwrap_model(model)

        safetensors_path = os.path.join(model_path, "adapter_model.safetensors")
        full_safetensors_path = os.path.join(model_path, "model.safetensors")
        bin_path = os.path.join(model_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            state_dict = _load_safetensors_robust(safetensors_path)
            # Also load auxiliary adapter weights (e.g. fake_score) if present
            aux_path = os.path.join(model_path, "adapter_model_aux.safetensors")
            if os.path.exists(aux_path):
                state_dict.update(_load_safetensors_robust(aux_path))
            missing, unexpected = unwrapped_model.load_state_dict(state_dict, strict=False)
            if missing:
                lora_missing = [k for k in missing if 'lora_' in k]
                tqdm.write(f"Loading model_{i}: {len(missing)} missing keys ({len(lora_missing)} LoRA)")
        elif os.path.exists(full_safetensors_path):
            state_dict = _load_safetensors_robust(full_safetensors_path)
            unwrapped_model.load_state_dict(state_dict)
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
            unwrapped_model.load_state_dict(state_dict)

    # Load optimizer(s)
    for i, optimizer in enumerate(accelerator._optimizers):
        optimizer_path = os.path.join(input_dir, f"optimizer_{i}.bin")
        if os.path.exists(optimizer_path):
            state_dict = torch.load(optimizer_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state_dict)

    # Load scheduler(s)
    for i, scheduler in enumerate(accelerator._schedulers):
        scheduler_path = os.path.join(input_dir, f"scheduler_{i}.bin")
        if os.path.exists(scheduler_path):
            state_dict = torch.load(scheduler_path, map_location="cpu", weights_only=False)
            scheduler.load_state_dict(state_dict)

    # Load random states
    random_states_path = os.path.join(input_dir, "random_states.bin")
    if os.path.exists(random_states_path):
        random_states = torch.load(random_states_path, map_location="cpu", weights_only=False)
        torch.set_rng_state(random_states["random_state"])
        np.random.set_state(random_states["numpy_random_state"])
        if torch.cuda.is_available() and "cuda_random_state" in random_states:
            torch.cuda.set_rng_state_all(random_states["cuda_random_state"])


def is_custom_checkpoint(checkpoint_dir: str) -> bool:
    """Check if a checkpoint was saved using custom_save_state."""
    # Custom checkpoints have optimizer_0.bin, standard accelerate checkpoints don't
    return os.path.exists(os.path.join(checkpoint_dir, "optimizer_0.bin"))

emo_cls = ['neutral', 'calm','happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
def format_time(second):
    s = int(second)
    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60)


class CustomCheckpoint:
    def __init__(
            self,
            save_first_step,
            save_freq_step,
            not_save_first_step_epoch,
            not_save_weight_only,
            ckpt_path,
            start_save_ep,
            save_freq,
            top_k,
            **kwargs
    ):
        self.save_first_step = save_first_step
        self.save_freq_step = save_freq_step
        self.save_weight_only = not not_save_weight_only
        self.save_first_step_epoch = not not_save_first_step_epoch
        self.ckpt_path = ckpt_path
        self.start_save_ep = start_save_ep
        self.save_freq = save_freq
        self.top_k = top_k
        self.prev_ckpts_step = glob(os.path.join(self.ckpt_path, "step-*"))
        self.prev_ckpts_epoch = glob(os.path.join(self.ckpt_path, "epoch-*"))
        self.prev_time = time.time()


    def on_train_start(self, trainer: Accelerator):
        if self.save_first_step:
            filename = os.path.join(self.ckpt_path, f'latest')
            trainer.wait_for_everyone()
            self._safe_save_state(trainer, filename)

            if trainer.is_main_process:
                message = f"Saving latest model to {filename}"
                tqdm.write(message)
                train_log = open(os.path.join(self.ckpt_path, f"{log_filename}"), 'a')
                train_log.write(message + '\n')
                train_log.close()

    def _safe_save_state(self, trainer: Accelerator, output_dir: str):
        """
        Save state with fallback to custom save when accelerator.save_state() fails
        due to optimizer/model count mismatch (e.g., DMD training with 3 models, 2 optimizers).
        """
        num_models = len(trainer._models)
        num_optimizers = len(trainer._optimizers)

        if num_optimizers != num_models:
            custom_save_state(trainer, output_dir)
        else:
            trainer.save_state(output_dir)


    def on_train_batch_end(self, trainer: Accelerator, global_step, batch_idx):
        if global_step > 2 and batch_idx % self.save_freq_step == 0 and (batch_idx > 0 or self.save_first_step_epoch):
            filename = os.path.join(self.ckpt_path, f'step-{global_step}')
            trainer.wait_for_everyone()
            self._safe_save_state(trainer, os.path.join(self.ckpt_path, 'latest'))

            if trainer.is_main_process:
                self.prev_ckpts_step.append(filename)
                if len(self.prev_ckpts_step) >= self.top_k:
                    import shutil
                    filename = self.prev_ckpts_step.pop(0)
                    if os.path.exists(filename):
                        shutil.rmtree(filename)

                curtime = time.time()
                interval = curtime - self.prev_time
                self.prev_time = curtime
                message = (f"***** Saving latest model to {os.path.abspath(filename)}, "
                           f"saving interval: {format_time(interval)} *****")
                tqdm.write(message)
                train_log = open(os.path.join(self.ckpt_path, f"{log_filename}"), 'a')
                train_log.write(message + '\n')
                train_log.close()


    def on_train_epoch_end(self, trainer: Accelerator, current_epoch):
        if current_epoch >= self.start_save_ep and current_epoch % self.save_freq == 0:
            filename = os.path.join(self.ckpt_path, f"epoch-{current_epoch}")
            trainer.wait_for_everyone()
            self._safe_save_state(trainer, filename)

            if trainer.is_main_process:
                self.prev_ckpts_epoch.append(filename)
                if len(self.prev_ckpts_epoch) >= self.top_k:
                    import shutil
                    filename = self.prev_ckpts_epoch.pop(0)
                    if os.path.exists(filename):
                        shutil.rmtree(filename)

                message = f"***** Saving latest model to {os.path.abspath(filename)} *****"
                tqdm.write(message)
                train_log = open(os.path.join(self.ckpt_path, f"{log_filename}"), 'a')
                train_log.write(message + '\n')
                train_log.close()


class ConsoleLogger:
    def __init__(
            self,
            print_freq,
            ckpt_path,
            batch_per_epoch,
            stage = "train",
            **kwargs
    ):
        self.step_frequcy = print_freq
        self.ckpt_path = ckpt_path
        self.batch_per_epoch = batch_per_epoch

        current_time = time.time()
        self.pre_iter_time = current_time
        self.pre_epoch_time = current_time
        self.total_time = 0

        self.stage = stage

    def on_train_batch_end(
            self,
            batch_idx,
            loss_dict,
            current_epoch,
            max_epoch,
            global_step,
            learning_rate,
            **kwargs
    ):
        if batch_idx % self.step_frequcy == 0 and batch_idx > 0:
            current_time = time.time()
            fmt_curtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
            iter_time = current_time - self.pre_iter_time
            self.total_time += iter_time
            self.pre_iter_time = current_time

            message = f"{fmt_curtime}"
            message += f", iter_time: {format_time(iter_time)}"
            message += f", total_time: {format_time(self.total_time)}"
            message += f", epoch: [{current_epoch}/{max_epoch}]"
            message += f", step: [{batch_idx}/{self.batch_per_epoch}]"
            message += f", global_step: {global_step}"
            message += f", lr: {learning_rate:.7f}"

            for label in loss_dict:
                message += f', {label}: {loss_dict[label]:.6f}'

            tqdm.write(message)
            train_log = open(os.path.join(self.ckpt_path, f"{log_filename}"), 'a')
            train_log.write(message + '\n')
            train_log.close()

    def on_train_epoch_end(self, current_epoch, global_step, learning_rate):
        current_time = time.time()
        iter_time = current_time - self.pre_iter_time
        epoch_time = current_time - self.pre_epoch_time
        self.total_time += iter_time
        self.pre_iter_time = current_time
        self.pre_epoch_time = current_time

        message = "{ "
        message += f"Epcoh {current_epoch} finished"
        message += f",\tglobal_step: {global_step}"
        message += f",\ttotal_time: {format_time(self.total_time)}"
        message += f",\tepoch_time: {format_time(epoch_time)}"
        message += f", current_lr: {learning_rate:.7f}"
        message += " }"

        tqdm.write(message)
        train_log = open(os.path.join(self.ckpt_path, f"{log_filename}"), 'a')
        train_log.write(message + '\n\n')
        train_log.close()


class ImageLogger:
    def __init__(
            self,
            ckpt_path,
            save_freq_step = 1,
            fps = 16,
            sample_num = None,
            clamp = True,
            increase_log_steps = True,
            batch_size = None,
            rescale = True,
            disabled = False,
            check_memory_use = False,
            log_on_batch_idx = True,
            log_first_step = True,
            step = 20,
            guidance_scale = 1.0,
            save_input = False,
            log_img_step = None,
            log_gif = False,
            deterministic_sample = True,
            accelerator = None,
            **kwargs
    ):
        self.fps = fps
        self.clamp = clamp
        self.rescale = rescale
        self.log_gif = log_gif
        self.save_input = save_input
        self.log_on_batch_idx = log_on_batch_idx
        self.disabled = disabled or sample_num == 0
        self.sample_num = default(sample_num, batch_size)
        self.batch_freq = default(log_img_step, save_freq_step)
        self.save_path = os.path.join(ckpt_path, kwargs.pop("mode"))
        self.log_first_step = log_first_step if not check_memory_use else False
        self.deterministic_sample = deterministic_sample
        self.sample_batch = None
        self.accelerator = accelerator

        self.sampling_params = {
            "unconditional_guidance_scale": guidance_scale,
            "step": step,
            "seed": torch.seed(),
        }

        # if load_checkpoint is None and pretrained is None and not check_memory_use:
        #     self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        # else:
        #     self.log_steps = []
        self.log_steps = []
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]


    def log_local(self, images, global_step, current_epoch, batch_idx, is_train):
        def save_image(img, path):
            if self.rescale:
                img = (img + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            img = img.permute(1, 2, 0).squeeze(-1)
            img = img.numpy()
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).save(path)

        def save_gif(imgs, path):
            if self.rescale:
                imgs = (imgs + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            imgs = imgs.permute(0, 2, 3, 1).squeeze(-1)
            imgs = imgs.numpy()
            imgs = (imgs * 255).astype(np.uint8)
            imgs = [Image.fromarray(img) for img in imgs]
            imgs[0].save(
                path,
                append_images=imgs[1:],
                save_all=True,
                duration=1000//self.fps,
                loop=0,
            )

        wandb_logs = {}
        for k in images:
            dirpath = os.path.join(self.save_path, k)
            os.makedirs(dirpath, exist_ok=True)

            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu().float()
                if len(images[k].shape) != 2:
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
            if len(images[k].shape) == 3:
                images[k] = images[k].unsqueeze(0)

            if is_train:
                # save gifs during training
                if self.log_gif:
                    if len(images[k].shape) == 2:
                        emo_id = int(images[k].squeeze(0))
                        filename = f"gs-{global_step:06}_e-{current_epoch:02}_b-{batch_idx:06}_emo_{emo_cls[emo_id]}.png"
                        path = os.path.join(dirpath, filename)
                        img = torch.zeros((3, 128, 128))
                        save_image(img, path)
                    else:
                        filename = f"gs-{global_step:06}_e-{current_epoch:02}_b-{batch_idx:06}.gif"
                        path = os.path.join(dirpath, filename)
                        save_gif(images[k], path)
                        # prepare wandb video log - images[k] is already (T, C, H, W)
                        imgs_np = images[k]
                        if self.rescale:
                            imgs_np = (imgs_np + 1.0) / 2.0
                        imgs_np = (imgs_np * 255).numpy().astype(np.uint8)  # (T, C, H, W)
                        # Resize for wandb display (fit within max_size to avoid scrolling)
                        T, C, H, W = imgs_np.shape
                        max_size = 384  # max dimension for both width and height
                        if H > max_size or W > max_size:
                            scale = min(max_size / H, max_size / W)
                            new_h, new_w = int(H * scale), int(W * scale)
                            resized_frames = []
                            for frame in imgs_np:
                                # frame is (C, H, W), convert to PIL, resize, convert back
                                frame_hwc = np.transpose(frame, (1, 2, 0))
                                pil_frame = Image.fromarray(frame_hwc)
                                pil_frame = pil_frame.resize((new_w, new_h), Image.LANCZOS)
                                resized_frames.append(np.transpose(np.array(pil_frame), (2, 0, 1)))
                            imgs_np = np.stack(resized_frames)
                        wandb_logs[f"samples/{k}"] = wandb.Video(imgs_np, fps=self.fps, format="gif")
                else:
                    grid = torchvision.utils.make_grid(images[k], nrow=4)
                    filename = f"gs-{global_step:06}_e-{current_epoch:02}_b-{batch_idx:06}.png"
                    path = os.path.join(dirpath, filename)
                    save_image(grid, path)
                    # prepare wandb image log
                    grid_np = grid
                    if self.rescale:
                        grid_np = (grid_np + 1.0) / 2.0
                    grid_np = grid_np.permute(1, 2, 0).numpy()
                    grid_np = (grid_np * 255).astype(np.uint8)
                    # Resize for wandb display (fit within max_size to avoid scrolling)
                    h, w = grid_np.shape[:2]
                    max_size = 384  # max dimension for both width and height
                    if h > max_size or w > max_size:
                        scale = min(max_size / h, max_size / w)
                        new_h, new_w = int(h * scale), int(w * scale)
                        pil_img = Image.fromarray(grid_np)
                        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
                        grid_np = np.array(pil_img)
                    wandb_logs[f"samples/{k}"] = wandb.Image(grid_np)

            else:
                # save images separately during testing
                filename = f"{batch_idx}.gif"
                path = os.path.join(dirpath, filename)
                save_gif(images[k], path)

        # log to wandb
        if is_train and wandb_logs and self.accelerator is not None:
            self.accelerator.log(wandb_logs, step=global_step)


    def log_img(self, model, batch, batch_idx, global_step=None, current_epoch=None):
        is_train = model.training
        if (hasattr(model, "log_images") and callable(model.log_images) and self.sample_num > 0) or not is_train:
            # self.sample_batch = self.sample_batch if self.sample_batch else batch
            if is_train:
                model.eval()
            # All ranks must run log_images to participate in DDP collective ops
            images = model.log_images(
                N = min(self.sample_num, MAXM_SAMPLE_SIZE) if is_train else self.sample_num,
                batch = batch, #self.sample_batch,
                return_inputs = is_train or self.save_input,
                **self.sampling_params,
            )

            # Only main process should log to disk/wandb
            is_main = self.accelerator is None or self.accelerator.is_main_process
            if is_main:
                self.log_local(images, global_step, current_epoch, batch_idx, is_train)

            if is_train:
                model.train()
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, model, global_step, current_epoch, batch, batch_idx, **kwargs):
        check_idx = batch_idx if self.log_on_batch_idx else global_step
        if not self.disabled and global_step > 0 and self.check_frequency(check_idx):
            if self.deterministic_sample:
                if self.sample_batch is None:
                    self.sample_batch = batch
                self.log_img(model, self.sample_batch, batch_idx, global_step, current_epoch)
                for k in self.sample_batch:
                    if isinstance(k, torch.Tensor):
                        self.sample_batch[k].cpu()
            else:
                self.log_img(model, batch, batch_idx, global_step, current_epoch)

    def on_test_batch_end(self, model, batch, batch_idx):
        self.log_img(model, batch, batch_idx)