import math
import torch

from einops import rearrange
from .base import Trainer as BaseTrainer
from ...util import exists, zero_drop


class Trainer(BaseTrainer):
    def setup_training(self):
        for m in self.model_list:
            m.requires_grad_(False).eval()
        self.model.flame_adapter = self.flame_adapter
        self.model.pose_patch_embedding.requires_grad_(True).train()
        self.model.flame_adapter.requires_grad_(True).train()
        self.model.motion_encoder.requires_grad_(True).train()
        self.model.enable_selective_grad()

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
            drop_face = zero_drop(xf, self.ucg_face_rate)
        
        motion_vec = self.flame_adapter(xmotion).to(self.dtype)
        conditions = {
            "context": [_ for _ in text_context],
            "seq_len": seq_len,
            "clip_fea": clip_context.to(self.dtype),
            "y": [_ for _ in y.detach()],
            "control": pose_latents,
            "face_pixel_values": xf * drop_face,
            "motion_vec": motion_vec * drop_face.view(b, 1, 1),
        }
        out = [torch.cat([ref_latents, latent], dim=2), conditions]
        if return_inputs: 
            out.extend([x, xc, xf, xt, xs])
        return out