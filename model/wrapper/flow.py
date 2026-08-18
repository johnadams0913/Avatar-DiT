import torch

from tqdm import tqdm
from typing import Optional, Tuple
from diffusers import FlowMatchEulerDiscreteScheduler

from model.util import exists, append_dims
from ..scheduler import FlowUniPCMultistepScheduler, FlowMatchScheduler
from .base_model import BaseModel



negative_text_cn = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
                    "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
                    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")


class FlowMatchingModel(BaseModel):

    def __init__(
            self,
            scheduler_config,
            *args,
            **kwargs
    ):
        self.register_scheduler(**scheduler_config)
        super().__init__(*args, **kwargs)

    def register_scheduler(
            self,
            num_train_timesteps: int = 1000,
            num_denoising_steps: int = None,
            scheduler_type: str = "normal",
            shift: float = 1.0,
            **scheduler_config,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        if scheduler_type == "normal":
            self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps, **scheduler_config)
        elif scheduler_type == "unipc":
            self.scheduler = FlowUniPCMultistepScheduler(num_train_timesteps, shift=shift, **scheduler_config)
        elif scheduler_type == "flow":
            self.scheduler = FlowMatchScheduler(num_train_timesteps, shift=shift, **scheduler_config)
        else:
            raise ValueError(f"Unsupported scheduler_type: {scheduler_type}. Choose from ['linear', 'euler', 'unipc']")

        self.num_denoising_steps = num_denoising_steps
        self.denoising_timesteps = None
        if exists(num_denoising_steps):
            step_size = num_train_timesteps // num_denoising_steps
            indices = torch.arange(1, num_denoising_steps + 1) * step_size - 1  # [249, 499, 749, 999]
            self.denoising_sigmas = self.scheduler.sigmas[indices]
            self.denoising_timesteps = self.scheduler.timesteps[indices]
            self.num_denoising_steps = num_denoising_steps

    def get_sigmas(self, timesteps):
        sigmas = self.scheduler.sigmas.to(timesteps.device)
        schedule_timesteps = self.scheduler.timesteps.to(timesteps.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        return sigmas[step_indices]


    def add_noise(
            self,
            x_start: torch.Tensor,
            noise: torch.Tensor,
            sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        sigma_t = append_dims(sigma_t, x_start.ndim).to(x_start.device).to(x_start.dtype)
        return (1 - sigma_t) * x_start + sigma_t * noise


    def _slice_temporal_conditions(
        self,
        cond_dict: dict,
        frame_index: int,
        frame_length: int,
        y_reft: Optional[torch.Tensor] = None
    ):
        sliced_cond = {}
        lat_t = frame_length // 4 + 1

        for key, value in cond_dict.items():
            if value is None:
                sliced_cond[key] = None
            elif key == "control":
                if exists(value):
                    value = value[0,:,frame_index:frame_index+frame_length]
                    sliced_cond[key] = torch.stack(self.vae.encode([value])).to(self.dtype)
                else:
                    sliced_cond[key] = None
            elif key in ["motion_vec", "face_pixel_values", "camera_params"]:
                sliced_cond[key] = value[:,frame_index:frame_index + frame_length]
            elif key == "y":
                sliced_y = []
                
                for i, v in enumerate(value):
                    lat_h, lat_w = v.shape[-2:]
                    H, W = lat_h * 8, lat_w * 8

                    if frame_index == 0 or not exists(y_reft):
                        # First segment: create zero-filled y_reft
                        y_reft = torch.stack(self.vae.encode([torch.zeros(3, frame_length, H, W, device=v.device, dtype=v.dtype)]))
                        mask_reft_len = 0
                        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, device=v.device)
                        y_reft = torch.concat([msk_reft, y_reft], dim=1)
                    else:
                        # Subsequent segments: use y_reft from previous generation
                        C_vid, overlap_frames, H_vid, W_vid = y_reft.shape
                        pad_frames = frame_length - overlap_frames
                        padded_video = torch.cat([
                            y_reft,
                            torch.zeros(C_vid, pad_frames, H_vid, W_vid, device=y_reft.device, dtype=y_reft.dtype)
                        ], dim=1)
                        padded_video = padded_video.to(v.device)
                        y_reft = torch.stack(self.vae.encode([padded_video]))
                        msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, overlap_frames, device=y_reft.device)
                        y_reft = torch.concat([msk_reft, y_reft], dim=1)

                    sliced_y.append(torch.concat([v, y_reft[0]], dim=1).to(self.dtype))

                sliced_cond[key] = sliced_y
            else:
                sliced_cond[key] = value
        return sliced_cond

    def sample(
            self,
            cfg_scale,
            cond: Optional[dict] = None,
            uncond: Optional[dict] = None,
            shape: Optional[Tuple[int, ...]] = None,
            num_steps: int = 50,
            shift: float = 5.0,
            sampler_type: str = "unipc",
            seed: int = -1,
            offload_model = False,
            frame_length: int = 21,
            chunk_num: int = 1,
            reference_frames: Optional[torch.Tensor] = 1,
            reft_len: int = None,
            **kwargs
    ) -> list[torch.Tensor]:
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
        reft_len = self.max_reft_len if not exists(reft_len) else reft_len

        if sampler_type == "unipc":
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps = self.num_train_timesteps,
                shift = shift,
                use_dynamic_shifting = False
            )
        elif sampler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps = self.num_train_timesteps,
                shift = shift,
                use_dynamic_shifting = False
            )
        else:
            raise NotImplementedError(f"Sampler {sampler_type} is not implemented.")

        frame_index = 0
        videos = []
        lat_t = frame_length // 4 + 1

        shape = (1, self.model.out_dim, lat_t + reference_frames, shape[0], shape[1])
        y_reft = None

        for chunk_idx in range(chunk_num):
            if exists(self.denoising_timesteps):
                timesteps = self.denoising_timesteps.to(device)
            else:
                if sampler_type == "unipc":
                    scheduler.set_timesteps(num_steps, shift=shift)
                elif sampler_type == "euler":
                    scheduler.set_timesteps(num_steps)
                timesteps = scheduler.timesteps.to(device)

            latent_slice = torch.stack([u for u in torch.randn(shape, dtype=self.dtype, generator=seed_g, device=device)])
            uncond_sliced = uncond
            cond_sliced = self._slice_temporal_conditions(cond, frame_index, frame_length, y_reft)
            if cfg_scale != 1.:
                uncond_sliced = self._slice_temporal_conditions(uncond, frame_index, frame_length, y_reft)

            for idx, t in enumerate(tqdm(timesteps)):
                latent_model_input = [l.cuda() for l in latent_slice]
                timestep = torch.stack([t]).cuda()

                noise_pred_cond = torch.stack(self.model(
                    latent_model_input, t=timestep, **cond_sliced
                )).to(torch.device('cpu') if offload_model else device)

                if offload_model:
                    torch.cuda.empty_cache()

                if cfg_scale != 1.:
                    noise_pred_uncond = torch.stack(self.model(
                        latent_model_input, t=timestep, **uncond_sliced
                    )).to(torch.device('cpu') if offload_model else device)

                    if offload_model:
                        torch.cuda.empty_cache()
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = noise_pred_cond

                latent_slice = latent_slice.to(torch.device('cpu') if offload_model else device)
                if exists(self.denoising_timesteps):
                    sigma = self.denoising_sigmas[idx:idx+1].to(latent_slice.device, latent_slice.dtype)
                    if idx < self.num_denoising_steps - 1:
                        sigma_next = self.denoising_sigmas[idx+1:idx+2].to(latent_slice.device, latent_slice.dtype)
                    else:
                        sigma_next = torch.zeros_like(sigma)
                    latent_slice = latent_slice + (sigma_next - sigma) * noise_pred
                else:
                    latent_slice = scheduler.step(
                        noise_pred,
                        t,
                        latent_slice,
                        return_dict=False,
                        generator=seed_g
                    )[0]

            if offload_model:
                torch.cuda.empty_cache()

            out_frames = self.vae.decode(
                [l for l in latent_slice[:,:,reference_frames:]]
            )[0].cpu()

            if frame_index > 0:
                out_frames = out_frames[:, reft_len:]
            videos.append(out_frames)

            frame_index += frame_length - reft_len
            if chunk_idx < chunk_num - 1:
                y_reft = out_frames[:, -reft_len:]
        return videos