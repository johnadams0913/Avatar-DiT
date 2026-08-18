import torch
import math

from einops import rearrange
from ..util import exists, rank_zero_print
from ..ray_sampler import get_camera_embedding
from .flow import FlowMatchingModel


class WanInferenceWrapper(FlowMatchingModel):
    def prepare_conditions(
            self,
            control,
            xc,
            face,
            flame,
            prompt,
            negative_prompt,
            frame_length,
            low_vram = False,
            **kwargs
    ):
        if low_vram:
            self.cpu()
            self.low_vram_shift(["vae", "img_embedder", "txt_embedder", "control_encoder"])

        control, xc, face = map(
            lambda t: rearrange(
                t.unsqueeze(0).cuda(), "b f c h w -> b c f h w"
            ).to(self.dtype) if exists(t) else None,
            (control, xc, face)
        )
        
        ref_latents, ic, tc, ntc = map(
            lambda t: t[0].encode(t[1]) if exists(t[1]) else None,
            (
                (self.vae, xc[:, :, :1]),
                (self.img_embedder, xc[:, :, :1]),
                (self.txt_embedder, prompt),
                (self.txt_embedder, negative_prompt)
            )
        )
        
        ref_latents = torch.stack(ref_latents).to(self.dtype).detach()
        
        lat_t = frame_length // 4 + 1
        b, _, _, lat_h, lat_w = ref_latents.shape
        
        if exists(ref_latents):
            mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=self.device)
            y_ref = torch.concat([mask_ref.repeat_interleave(b, dim=0), ref_latents], dim=1)
        else:
            # Create zero ref_latents if not available
            mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=self.device)
            zero_ref = torch.zeros(b, 4, 1, lat_h, lat_w, device=self.device, dtype=self.dtype)
            y_ref = torch.concat([mask_ref.repeat_interleave(b, dim=0), zero_ref], dim=1)

        y = y_ref.to(self.dtype).detach()

        seq_len = math.ceil(
            (lat_h * lat_w) /
            (self.model.patch_size[1] * self.model.patch_size[2]) *
            (lat_t + 1)
        )

        if exists(ic):
            ic = ic.to(self.dtype).cuda()
        y = y.to(self.dtype).cuda()
        tc, ntc = map(lambda ts: [t.to(self.dtype).cuda() for t in ts], (tc, ntc))
        
        # Process flame data with flame_adapter to get motion_vec
        if exists(flame):
            rank_zero_print(f"flame.shape: {flame.shape}")
            adapter_param = next(self.flame_adapter.parameters())
            flame = flame.unsqueeze(0).to(device=adapter_param.device, dtype=adapter_param.dtype)
            motion_vec = self.flame_adapter(flame).to(device=self.device, dtype=self.dtype)
        else:
            motion_vec = None

        if exists(control):
            rank_zero_print(f"[DEBUG] prepare_conditions: control.shape={control.shape}")
        if exists(motion_vec):
            rank_zero_print(f"[DEBUG] prepare_conditions: motion_vec.shape={motion_vec.shape}")
        
        cond = {
            "context": tc,
            "clip_fea": ic,
            "seq_len": seq_len,
            "y": [t for t in y],
            "control": control,
            # "face_pixel_values": None,
            "face_pixel_values": face,
            "motion_vec": motion_vec,
        }

        uncond = cond.copy()
        uncond.update({"context": ntc, "motion_vec": None})
        return cond, uncond


class MultiViewInferenceWrapper(WanInferenceWrapper):
    def prepare_conditions(
            self,
            control,
            xc,
            face,
            flame,
            prompt,
            camera,
            frame_length,
            negative_prompt,
            low_vram = False,
            reference_num = 1,
            **kwargs
    ):
        if low_vram:
            self.cpu()
            self.low_vram_shift(["vae", "img_embedder", "txt_embedder", "control_encoder"])

        control, xc, face = map(
            lambda t: rearrange(
                t.unsqueeze(0).cuda(), "b f c h w -> b c f h w"
            ).to(self.dtype) if exists(t) else None,
            (control, xc, face)
        )
        reference_num = xc.size(2)

        ic = self.img_embedder.encode(xc[:, :, :1]) if exists(xc) else None
        tc = self.txt_embedder.encode(prompt)
        ntc = self.txt_embedder.encode(negative_prompt)

        if exists(xc):
            ref_latents = torch.cat(
                [torch.stack(self.vae.encode(xc[:, :, idx:idx+1])) for idx in range(reference_num)],
                dim=2
            )
        else:
            ref_latents = None

        ref_latents = ref_latents.to(self.dtype).detach() if exists(ref_latents) else None
        
        lat_t = frame_length // 4 + 1
        lat_h, lat_w = ref_latents.shape[3:]
        
        mask_ref = self.get_i2v_mask(reference_num, lat_h, lat_w, 1, device=self.device)
        y_ref = torch.concat([mask_ref, ref_latents], dim=1)

        y = y_ref.to(self.dtype).detach()
        seq_len = math.ceil(
            (lat_h * lat_w) /
            (self.model.patch_size[1] * self.model.patch_size[2]) *
            (lat_t + reference_num)
        )

        if exists(ic):
            ic = ic.to(self.dtype).cuda()
        y = y.to(self.dtype).cuda()
        tc, ntc = map(lambda ts: [t.to(self.dtype).cuda() for t in ts], (tc, ntc))
        
        # Process flame data with flame_adapter to get motion_vec
        if exists(flame):
            adapter_param = next(self.flame_adapter.parameters())
            flame = flame.unsqueeze(0).to(device=adapter_param.device, dtype=adapter_param.dtype)
            motion_vec = self.flame_adapter(flame).to(device=self.device, dtype=self.dtype)
        else:
            motion_vec = None
        
        # Process camera parameters
        if exists(camera):
            camera = camera.cuda().to(self.dtype)
            camera = get_camera_embedding(camera, lat_t, lat_h // 2, lat_w // 2)

        cond = {
            "context": tc,
            "clip_fea": ic,
            "seq_len": seq_len,
            "y": [t for t in y],
            "control": control,
            "face_pixel_values": None,
            # "face_pixel_values": face,
            "motion_vec": motion_vec,
            "camera_params": camera,
        }

        uncond = cond.copy()
        uncond.update({"context": ntc})
        return cond, uncond