import math
import torch
import torch.nn.functional as F

from typing import Optional
from einops import rearrange

from ..flow import FlowMatchingModel
from ...util import exists, zero_drop, default


class Trainer(FlowMatchingModel):
    def __init__(
            self,
            gt_key,
            control_key,
            face_key = "face",
            depth_key = "depth",
            image_key = "reference",
            text_key = "text",
            camera_key = "camera",
            bg_key = None,
            mcode_key = None,
            prediction_outputs = False,
            ucg_face_rate = 0.,
            ucg_text_rate = 0.,
            ucg_img_rate = 0.,
            ucg_control_rate = 0.,
            lora_training = False,
            *args,
            **kwargs
    ):
        self.gt_key = gt_key
        self.control_key = control_key
        self.face_key = face_key
        self.depth_key = depth_key
        self.image_key = image_key
        self.text_key = text_key
        self.motion_key = mcode_key
        self.camera_key = camera_key
        self.prediction_outputs = prediction_outputs

        self.ucg_text_rate = ucg_text_rate
        self.ucg_img_rate = ucg_img_rate
        self.ucg_control_rate = ucg_control_rate
        self.ucg_face_rate = ucg_face_rate
        self.lora_training = lora_training
        
        super().__init__(*args, **kwargs)
        if not getattr(self, "_defer_setup_training", False):
            self.setup_training()

    def setup_training(self):
        for m in self.model_list:
            m.requires_grad_(False).eval()
        if self.lora_training:
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)
        else:
            self.model.pose_patch_embedding.requires_grad_(True).train()

    def get_trainable_params(self):
        return list(filter(lambda p: p.requires_grad, self.model.parameters()))

    def on_train_start(self, accelerator):
        if accelerator.mixed_precision == 'no':
            self.dtype = torch.float32
        elif accelerator.mixed_precision == 'bf16':
            self.dtype = torch.bfloat16
        elif accelerator.mixed_precision == 'fp16':
            self.dtype = torch.float16
        else:
            raise ValueError(f"Invalid mixed precision {accelerator.mixed_precision}")

        for m in self.model_list:
            if exists(m):
                m.to(accelerator.device).to(self.half_precision_dtype)

    def on_train_batch_end(self):
        pass

    def training_step(self, batch):
        latent, cond = self.get_input(batch)
        loss = self.p_losses(latent, cond)
        return loss

    def p_losses(
            self,
            x_start: torch.Tensor,
            cond: Optional[dict] = None,
            mask: torch.Tensor = None
    ) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        target = x_start if self.prediction_outputs else noise - x_start

        # discrete timestep sampling
        t_index = torch.randint(0, self.num_train_timesteps, (x_start.shape[0],),
                          device=self.scheduler.timesteps.device)
        timestep = self.scheduler.timesteps[t_index].to(x_start.device)
        sigmas_t = self.get_sigmas(timestep)
        x_noisy = self.add_noise(x_start, noise, sigmas_t)
        model_output = self.model([_ for _ in x_noisy], timestep, **cond)
        model_output = torch.stack(model_output)

        # Compute loss
        mask = default(mask, 1.0)
        loss = F.mse_loss(model_output.float(), target.float(), reduction="mean") * mask
        return loss


    def get_input(
            self,
            batch,
            bs = None,
            return_inputs = False,
            **kwargs
    ) -> list:
        with torch.no_grad():
            if exists(bs):
                for k in batch:
                    batch[k] = batch[k][:bs]

            x = batch[self.gt_key]
            xc = batch[self.image_key]
            xt = batch[self.text_key]
            xf = batch.get(self.face_key, None)
            xs = batch.get(self.control_key, None)
            # xs = None
            xbg = batch.get("bg", None)
            xmask = batch.get("mask", None)
            xmotion = batch.get(self.motion_key, None)

            x, xc, xf, xs, xbg = map(
                lambda t: rearrange(
                    t.to(memory_format=torch.contiguous_format).to(self.dtype),
                    "b f c h w -> b c f h w"
                ) if exists(t) else None, (x, xc, xf, xs, xbg)
            )

            if exists(xmask):
                xmask = rearrange(
                    xmask.to(memory_format=torch.contiguous_format).to(self.dtype),
                    "b f c h w -> b c f h w"
                )

            if exists(xmotion):
                xmotion = xmotion.to(memory_format=torch.contiguous_format).to(self.dtype)

            b, c, f, h, w = x.shape
            latent, pose_latents, ref_latents, clip_context, text_context = map(
                lambda t: t[0].encode(t[1]) if exists(t[1]) else None,
                (
                    (self.vae, x),
                    (self.vae, xs if exists(xs) else None),
                    (self.vae, xc[:, :, :1]),
                    (self.img_embedder, xc[:, :, :1]),
                    (self.txt_embedder, xt)
                ),
            )
            latent, pose_latents, ref_latents, text_context = map(
                lambda t: torch.stack(t).to(self.dtype).detach() if exists(t) else None,
                (latent, pose_latents, ref_latents, text_context)
            )
            lat_t, lat_h, lat_w = latent.shape[2:]
            mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=self.device)
            y_ref = torch.concat([mask_ref.repeat_interleave(b, dim=0), ref_latents], dim=1)

            mask_reft_len = self.max_reft_len
            if mask_reft_len == 0:
                y_reft = torch.stack(self.vae.encode([torch.zeros(3, f, h, w, device=x.device, dtype=self.dtype)]))
            else:
                context_frames = x[0, :, :mask_reft_len]
                y_reft = torch.stack(self.vae.encode([torch.cat([
                    context_frames,
                    torch.zeros(3, f-mask_reft_len, h, w, device=x.device, dtype=self.dtype)
                ], dim=1)]))

            msk_reft = self.get_i2v_mask(lat_t, lat_h, lat_w, mask_reft_len, device=x.device)
            y_reft = torch.concat([msk_reft.repeat_interleave(b, dim=0), y_reft], dim=1)
            y = torch.concat([y_ref, y_reft], dim=2).to(self.dtype).detach()

            seq_len = math.ceil(
                (lat_h * lat_w) /
                (self.model.patch_size[1] * self.model.patch_size[2]) *
                (lat_t + 1)
            )

            text_context = text_context * zero_drop(text_context, self.ucg_text_rate)
            clip_context = clip_context * zero_drop(clip_context, self.ucg_img_rate)
            pose_latents = pose_latents * zero_drop(pose_latents, self.ucg_control_rate) if exists(pose_latents) else None
            
        conditions = {
            "context": [_ for _ in text_context],
            "seq_len": seq_len,
            "clip_fea": clip_context.to(self.dtype),
            "y": [_ for _ in y.detach()],
            "control": pose_latents,
        }
        out = [torch.cat([ref_latents, latent], dim=2), conditions]
        if return_inputs: 
            out.extend([x, xc, xf, xt, xs])
        return out

    @torch.no_grad()
    def log_images(
            self,
            batch,
            N = 4,
            step = 20,
            unconditional_guidance_scale = 9.0,
            return_inputs = False,
            seed = None,
            **kwargs
    ):
        out = self.get_input(
            batch,
            bs=N,
            return_inputs=return_inputs,
            **kwargs
        )
        if exists(seed):
            torch.manual_seed(seed)

        log = dict()
        if return_inputs:
            z, c = out[:2]
            x, xc, xf, xt, xs = [
                rearrange(t, "b c f h w -> (b f) c h w").contiguous() if exists(t) and isinstance(t, torch.Tensor)
                else t for t in out[2:]
            ]

            log["inputs"] = x
            log["conditioning"] = xc
            log["reconstruction"] = rearrange(
                torch.stack(self.vae.decode(z[:, :, 1:].to(self.dtype))),
                "b c f h w -> (b f) c h w"
            )

            if exists(xs):
                log["control"] = (xs[:, :3] - 0.5) / 0.5
            if exists(xf):
                log["face"] = (xf - 0.5) / 0.5
        else:
            z, c = out

        x_T = torch.randn_like(z)
        shape = x_T.shape[-2:]
        lat_t = z.shape[2] - 1
        samples_cfg = self.sample(
            cond = c,
            num_steps = step,
            shape = shape,
            # uncond = uc_full,
            cfg_scale = unconditional_guidance_scale,
            latent = x_T,
            device = z.device,
            frame_length = (lat_t - 1) * 4 + 1,
            video_length = 0,
            **kwargs
        )
        samples_cfg = rearrange(torch.stack(samples_cfg), "b c f h w -> (b f) c h w")
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = samples_cfg

        if exists(seed):
            torch.seed()
        return log

    def _slice_temporal_conditions(
        self, 
        cond_dict: dict,
        *args,
        **kwargs
    ):
        return cond_dict