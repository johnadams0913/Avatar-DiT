"""
    Copied from Uni3C's official implementation.
    Uni3C: Unifying Precisely 3D-Enhanced Camera and Human Motion Controls for Video Generation
    https://github.com/alibaba-damo-academy/Uni3C/blob/main/src/camera.py
"""

import einops
import torch
import torch.nn.functional as F


@torch.amp.autocast("cuda", enabled=False)
def batch_sample_rays(intrinsic, extrinsic, image_h=None, image_w=None):
    ''' get rays
    Args:
        intrinsic: [BF, 3, 3],
        extrinsic: [BF, 4, 4],
        h, w: int
        # normalize: let the first camera R=I
    Returns:
        rays_o, rays_d: [BF, N, 3]
    '''

    device = intrinsic.device
    B = intrinsic.shape[0]

    extrinsic_f32 = extrinsic.to(dtype=torch.float32)
    intrinsic_f32 = intrinsic.to(dtype=torch.float32)
    c2w = torch.inverse(extrinsic_f32)[:, :3, :4].to(device)  # [BF,3,4]
    x = torch.arange(image_w, device=device).float() - 0.5
    y = torch.arange(image_h, device=device).float() + 0.5
    points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
    points = einops.repeat(points, 'w h c -> b (h w) c', b=B)
    points = torch.cat([points, torch.ones_like(points)[:, :, 0:1]], dim=-1)
    directions = points @ intrinsic_f32.inverse().to(device).transpose(-1, -2) * 1  # depth is 1

    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)  # [BF,N,3]
    rays_o = c2w[..., :3, 3]  # [BF, 3]

    rays_o = rays_o[:, None, :].expand_as(rays_d)  # [BF, N, 3]

    return rays_o, rays_d


@torch.amp.autocast("cuda", enabled=False)
def embed_rays(rays_o, rays_d, nframe):
    if len(rays_o.shape) == 4:  # [b,f,n,3]
        rays_o = einops.rearrange(rays_o, "b f n c -> (b f) n c")
        rays_d = einops.rearrange(rays_d, "b f n c -> (b f) n c")
    cross_od = torch.cross(rays_o, rays_d, dim=-1)
    cam_emb = torch.cat([rays_d, cross_od], dim=-1)
    cam_emb = einops.rearrange(cam_emb, "(b f) n c -> b f n c", f=nframe)
    return cam_emb


def transform_plucker(plucker, w2c):
    """
    Apply world-to-camera transform to Plucker rays.
    plucker: [B, N, 6] or [BF, N, 6]
    w2c: [B, 4, 4] or [BF, 4, 4]
    """
    direction = plucker[..., :3]
    moment = plucker[..., 3:]
    R = w2c[..., :3, :3]
    t = w2c[..., :3, 3]

    direction_t = torch.matmul(R, direction.unsqueeze(-1)).squeeze(-1)
    moment_t = torch.matmul(R, moment.unsqueeze(-1)).squeeze(-1)
    moment_t = moment_t + torch.cross(t.unsqueeze(-2).expand_as(direction_t), direction_t, dim=-1)
    return torch.cat([direction_t, moment_t], dim=-1)


def relative_transform_plucker(plucker, ref_w2c, nframe=None):
    """
    Transform Plucker rays into reference camera coordinates.
    plucker: [B, F, N, 6] or [BF, N, 6]
    ref_w2c: [B, 4, 4] or [BF, 4, 4]
    """
    if plucker.dim() == 4:
        b, f, n, _ = plucker.shape
        plucker_flat = plucker.reshape(b * f, n, 6)
        if ref_w2c.dim() == 3:
            if ref_w2c.shape[0] == b and nframe is not None:
                ref_w2c = ref_w2c.repeat_interleave(nframe, dim=0)
            elif ref_w2c.shape[0] == b:
                ref_w2c = ref_w2c.repeat_interleave(f, dim=0)
        plucker_rel = transform_plucker(plucker_flat, ref_w2c)
        return plucker_rel.reshape(b, f, n, 6)
    return transform_plucker(plucker, ref_w2c)


@torch.amp.autocast("cuda", enabled=False)
def camera_center_normalization(w2c, nframe, camera_scale=2.0):
    # copy from SEVA, w2c: [BF, 4, 4]
    w2c_dtype = w2c.dtype
    if w2c_dtype in (torch.bfloat16, torch.float16):
        w2c = w2c.float()
    # ensure the first view is eye matrix
    c2w_view0 = w2c[::nframe].inverse()  # [B,4,4]
    c2w_view0 = c2w_view0.repeat_interleave(nframe, dim=0)  # [BF,4,4]
    w2c = c2w_view0 @ w2c

    # camera centering
    c2w = torch.linalg.inv(w2c)
    camera_dist_2med = torch.norm(c2w[:, :3, 3] - c2w[:, :3, 3].median(0, keepdim=True).values, dim=-1)
    valid_mask = camera_dist_2med <= torch.clamp(torch.quantile(camera_dist_2med, 0.97) * 10, max=1e6)
    c2w[:, :3, 3] -= c2w[valid_mask, :3, 3].mean(0, keepdim=True)
    w2c = torch.linalg.inv(c2w)

    # camera normalization
    camera_dists = c2w[:, :3, 3].clone()
    translation_scaling_factor = (
        camera_scale
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1, dtype=camera_dists.dtype, device=camera_dists.device),
            atol=1e-5,
        ).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    w2c[:, :3, 3] *= translation_scaling_factor
    c2w[:, :3, 3] *= translation_scaling_factor

    return w2c.to(w2c_dtype)


def get_camera_embedding(camera_params, f, h, w, normalize=True, return_rays=False):
    if camera_params.dim() == 3:
        b, nf, _ = camera_params.shape
        camera_params = camera_params.reshape(b * nf, -1)
    extrinsic = camera_params[:, :16].view(-1, 4, 4)
    intrinsic = camera_params[:, 16:].view(-1, 3, 3)
    extrinsic_h = extrinsic
    if normalize:
        extrinsic_h = camera_center_normalization(extrinsic_h, nframe=f)

    rays_o, rays_d = batch_sample_rays(intrinsic, extrinsic_h, image_h=h, image_w=w)
    camera_embedding = embed_rays(rays_o, rays_d, nframe=f)
    camera_embedding = einops.rearrange(camera_embedding, "b f (h w) c -> b c f h w", h=h, w=w)

    if return_rays:
        return camera_embedding, rays_o, rays_d
    return camera_embedding


def visualize_plucker_rays(
    plucker: torch.Tensor = None,
    rays_o: torch.Tensor = None,
    rays_d: torch.Tensor = None,
    num_rays: int = 128,
    ray_length: float = 0.2,
    seed: int = 0,
):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    if rays_o is not None and rays_d is not None:
        assert torch.is_tensor(rays_o) and torch.is_tensor(rays_d), "rays_o and rays_d must be torch tensors"
        assert rays_o.shape == rays_d.shape, "rays_o and rays_d must have the same shape"
        assert rays_o.shape[-1] == 3, "rays_o and rays_d must have last dim 3"
        
        batch_size = rays_o.shape[0]
        device = rays_o.device
        origins_list = []
        directions_list = []
        
        for b in range(batch_size):
            origins_list.append(rays_o[b].reshape(-1, 3))
            directions_list.append(rays_d[b].reshape(-1, 3))
            
    elif plucker is not None:
        assert torch.is_tensor(plucker), "plucker must be a torch tensor"
        assert plucker.shape[-1] == 6, "plucker must have last dim 6"
        
        if plucker.dim() == 2:
            plucker = plucker.unsqueeze(0)
        
        batch_size = plucker.shape[0]
        device = plucker.device
        origins_list = []
        directions_list = []
        
        for b in range(batch_size):
            rays = plucker[b].reshape(-1, 6)
            direction = rays[:, :3]
            moment = rays[:, 3:]
            d_norm_sq = (direction * direction).sum(dim=-1, keepdim=True) + 1e-8
            origin = torch.cross(direction, moment, dim=-1) / d_norm_sq
            origins_list.append(origin)
            directions_list.append(direction)
    else:
        raise ValueError("Must provide either (rays_o and rays_d) or plucker")

    img_tensors = []
    
    for b in range(batch_size):
        origin = origins_list[b]
        direction = directions_list[b]
        
        total = origin.shape[0]
        if total == 0:
            raise ValueError("No rays to visualize")

        gen = torch.Generator(device=device)
        gen.manual_seed(seed + b)
        num_rays_sample = min(num_rays, total)
        indices = torch.randperm(total, generator=gen, device=device)[:num_rays_sample]
        
        origin_sampled = origin.index_select(0, indices)
        direction_sampled = direction.index_select(0, indices)

        origin_np = origin_sampled.detach().cpu().numpy()
        dir_np = direction_sampled.detach().cpu().numpy()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        
        colors = cm.viridis(np.linspace(0, 1, num_rays_sample))
        for i in range(num_rays_sample):
            o = origin_np[i]
            d = dir_np[i]
            ax.plot(
                [o[0], o[0] + d[0] * ray_length],
                [o[1], o[1] + d[1] * ray_length],
                [o[2], o[2] + d[2] * ray_length],
                color=colors[i],
                linewidth=0.8,
                alpha=0.6,
            )

        unique_origins = np.unique(origin_np, axis=0)
        if len(unique_origins) <= 20:
            for cam_pos in unique_origins:
                ax.scatter(*cam_pos, color='red', s=100, marker='o', 
                          edgecolors='black', linewidths=1.5, zorder=10)
        else:
            cam_pos = origin_np.mean(axis=0)
            ax.scatter(*cam_pos, color='red', s=100, marker='o', 
                      edgecolors='black', linewidths=1.5, label='Camera Center', zorder=10)
            ax.legend()

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"Plucker Ray Visualization (Batch {b}, {num_rays_sample} rays)")
        ax.grid(True, alpha=0.3)
        ax.view_init(elev=20, azim=45)
        
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape((height, width, 4))
        img = img[:, :, :3]
        plt.close(fig)
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensors.append(img_tensor)
    
    result = torch.stack(img_tensors, dim=0)
    return result