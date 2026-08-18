import math
import torch

from einops import rearrange

from .base import Trainer
from ...util import exists, zero_drop
from ...ray_sampler import *


class MultiViewTrainer(Trainer):
    def __init__(
            self,
            mv_key = "otherview",
            *args,
            **kwargs    
    ):
        self.mv_key = mv_key
        super().__init__(*args, **kwargs)
        self.setup_training()

    def setup_training(self):
        for m in self.model_list:
            m.requires_grad_(False).eval()

        self.model.camera_embedding.requires_grad_(True).train()
        self.model.patch_embedding.requires_grad_(True).train()
        self.model.pose_patch_embedding.requires_grad_(True).train()

        if self.lora_training:
            self.model.train()
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)

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
            camera = batch.get(self.camera_key, None)

            xbg = batch.get("bg", None)
            xmask = batch.get("mask", None)
            xmotion = batch.get(self.motion_key, None)

            x, xc, xf, xs, xbg, xmask = map(
                lambda t: rearrange(
                    t.to(memory_format=torch.contiguous_format).to(self.dtype), 
                    "b f c h w -> b c f h w"
                ) if exists(t) else None,
                (x, xc, xf, xs, xbg, xmask)
            )
            xmotion, camera = map(lambda t: 
                t.to(memory_format=torch.contiguous_format).to(self.dtype) if exists(t) else None, 
                (xmotion, camera)
            )

            b, c, f, h, w = x.shape
            latent, pose_latents, clip_context, text_context = map(
                lambda t: t[0].encode(t[1]) if exists(t[1]) else None,
                (
                    (self.vae, x),
                    (self.vae, xs if exists(xs) else None),
                    (self.img_embedder, xc[:, :, :1]),
                    (self.txt_embedder, xt)
                ),
            )
            ref_latents = torch.cat([torch.stack(
                self.vae.encode(cl.unsqueeze(2))
            ) for cl in xc.unbind(dim=2)], dim=2)

            latent, pose_latents, text_context = map(
                lambda t: torch.stack(t).to(self.dtype).detach() if exists(t) else None,
                (latent, pose_latents, text_context)
            )
            lat_t, lat_h, lat_w = latent.shape[2:]
            additional_reference_frames = ref_latents.shape[2]

            mask_ref = self.get_i2v_mask(additional_reference_frames, lat_h, lat_w, 1, device=self.device)
            y_ref = torch.concat([mask_ref.repeat_interleave(b, dim=0), ref_latents], dim=1)
    
            if exists(camera) and not camera.all() == 0:
                camera, rays_o, rays_d = get_camera_embedding(camera, lat_t, lat_h//2, lat_w//2, return_rays=True)
            else:
                camera = None
                rays_o = None
                rays_d = None

            seq_len = math.ceil(
                (lat_h * lat_w) /
                (self.model.patch_size[1] * self.model.patch_size[2]) *
                (lat_t + additional_reference_frames)
            )

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
            
            text_context = text_context * zero_drop(text_context, self.ucg_text_rate)
            clip_context = clip_context * zero_drop(clip_context, self.ucg_img_rate)
            pose_latents = pose_latents * zero_drop(pose_latents, self.ucg_control_rate) if exists(pose_latents) else None
            drop_face = zero_drop(xf, self.ucg_face_rate) if exists(xf) else None
        
        motion_vec = None
        # motion_vec = self.model.flame_adapter(xmotion).to(self.dtype) if exists(xmotion) else None
        ref_latents = ref_latents.to(self.dtype)
        conditions = {
            "context": [_ for _ in text_context],
            "seq_len": seq_len,
            "clip_fea": clip_context.to(self.dtype),
            "y": [_ for _ in y],
            "control": pose_latents,
            "camera_params": None ,
            "face_pixel_values": xf * drop_face if exists(xf) else None,
            "motion_vec": motion_vec * drop_face.view(b, 1, 1) if exists(xf) else None,
        }
        out = [torch.cat([ref_latents, latent], dim=2), conditions]
        if return_inputs: 
            out.extend([x, xc, xf, xt, xs, rays_o, rays_d])
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
        additional_reference_frames = 1
        if return_inputs:
            z, c = out[:2]
            x, xc, xf, xt, xs, rays_o, rays_d = out[2:]
            frame_length = x.shape[2]
            additional_reference_frames = xc.shape[2] if exists(xc) else 1
            x, xc, xf, xs = [
                rearrange(t, "b c f h w -> (b f) c h w").contiguous() if exists(t) and isinstance(t, torch.Tensor)
                else t for t in [x, xc, xf, xs]
            ]
            
            log["inputs"] = x
            log["conditioning"] = xc
            
            if exists(rays_o) and exists(rays_d):
                log["camera"] = visualize_plucker_rays(rays_o=rays_o, rays_d=rays_d)
            log["reconstruction"] = rearrange(
                torch.stack(self.vae.decode(z[:, :, additional_reference_frames:].to(self.dtype))),
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

        samples_cfg = self.sample(
            cond = c,
            num_steps = step,
            shape = shape,
            # uncond = uc_full,
            cfg_scale = unconditional_guidance_scale,
            latent = x_T,
            device = z.device,
            frame_length = frame_length,
            video_length = 0,
            reference_frames = additional_reference_frames,
            **kwargs
        )
        samples_cfg = rearrange(torch.stack(samples_cfg), "b c f h w -> (b f) c h w")
        log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = samples_cfg

        if exists(seed):
            torch.seed()
        return log