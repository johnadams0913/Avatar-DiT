import PIL.Image as Image
import numpy as np
import numpy.random as random

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms


def exists(v):
    return v is not None


def to_tensor(x):
    return transforms.ToTensor()(x)


def normalize(img, grayscale=False, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    img = to_tensor(img)
    if grayscale:
        img = transforms.Normalize((0.5), (0.5))(img)
    else:
        img = transforms.Normalize(mean, std)(img)
    return img

def track_crop(img, mean_center, target_size=(512, 768)):
    nw, nh = target_size
    w, h = img.size

    target_aspect = nw / nh
    img_aspect = w / h

    if img_aspect > target_aspect:
        ratio = nh / h
    else:
        ratio = nw / w

    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))
    img = img.resize((new_w, new_h))
    w, h = new_w, new_h

    cx = int(mean_center[0] * ratio)
    cy = int(mean_center[1] * ratio)

    left = cx - nw // 2
    right = left + nw
    top = cy - nh // 2
    bottom = top + nh

    if left < 0:
        left, right = 0, nw
    elif right > w:
        left, right = w - nw, w

    if top < 0:
        top, bottom = 0, nh
    elif bottom > h:
        top, bottom = h - nh, h

    return img.crop((max(left, 0), max(top, 0), min(right, w), min(bottom, h)))


def random_erase(img: Image, max_num = 1500, min_grid_size = 3, max_grid_size = 20):
    """
    Randomly erases (sets to white) square patches in an image with random grid sizes.

    Args:
        max_num (int): Maximum number of patches to erase.
        min_grid_size (int): Minimum size of each square patch.
        max_grid_size (int): Maximum size of each square patch.

    Returns:
        np.ndarray: Image with random erasures applied.
    """
    img = np.array(img)
    h, w, c = img.shape

    num = random.randint(1, max_num)  # Number of patches to erase

    # Apply erasure
    for _ in range(num):
        # Randomly determine grid size
        grid_size = random.randint(min_grid_size, max_grid_size)

        # Ensure grid positions are valid within the image dimensions
        x_start = random.randint(0, max(0, w - grid_size))
        y_start = random.randint(0, max(0, h - grid_size))

        # Set the patch to white (255)
        img[y_start:y_start + grid_size, x_start:x_start + grid_size, :] = 255

    return img


def random_erase_gaussian(img, max_num = 50, grid_size = 64, mean = 0.5, stddev = 0.1):
    """
    Randomly erases (sets to white) square patches in an image with random grid sizes.

    Args:
        img (np.ndarray): Input image as a NumPy array (H, W, C).
        max_num (int): Maximum number of patches to erase.
        min_grid_size (int): Minimum size of each square patch.
        max_grid_size (int): Maximum size of each square patch.

    Returns:
        np.ndarray: Image with random erasures applied.
    """
    h, w, c = img.shape

    # Ensure the input is in uint8 format
    img = img.astype(np.uint8)

    # Randomly select grid regions
    num = random.randint(1, max_num)  # Number of grid regions to mask

    for _ in range(num):
        # Randomly choose grid region
        x_start = random.randint(0, max(0, w - grid_size))
        y_start = random.randint(0, max(0, h - grid_size))

        # Generate binary mask based on a Gaussian distribution
        mask = np.random.normal(mean, stddev, (grid_size, grid_size))
        mask = (mask > 0.5).astype(np.uint8)  # Threshold to create a binary mask (0 or 1)

        # Apply the mask to the grid region
        for i in range(c):  # For each channel
            region = img[y_start:y_start + grid_size, x_start:x_start + grid_size, i]
            region[mask == 0] = 255  # Set pixels to white where mask is 0

    return img


def compute_output_padding(original_size, kernel_size, stride, downsampled_size):
    return original_size - ((downsampled_size - 1) * stride + kernel_size)


def mask_expansion(mask: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    """
    Rasterize a mask tensor into grids of specified shape.

    This function downsamples a mask using max pooling with the specified grid dimensions,
    then upsamples it back to the original size using transposed convolution.

    Args:
        mask: Input mask tensor with shape (batch, channels, height, width) or (channels, height, width)
        grid_h: Height of each grid cell
        grid_w: Width of each grid cell

    Returns:
        torch.Tensor: Rasterized mask with same dimensions as input

    Raises:
        ValueError: If grid dimensions are not positive integers or mask has invalid shape
    """
    # Validate inputs
    if not isinstance(grid_h, int) or not isinstance(grid_w, int) or grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"Grid dimensions must be positive integers, got ({grid_h}, {grid_w})")

    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"Mask must be a torch.Tensor, got {type(mask)}")

    if len(mask.shape) not in [3, 4]:
        raise ValueError(f"Mask must have 3 or 4 dimensions, got shape {mask.shape}")

    # Handle 3D input (single batch)
    if len(mask.shape) == 3:
        reshape = True
        _, h, w = mask.shape
        mask = mask.unsqueeze(0)  # Add batch dimension
    else:
        reshape = False
        _, _, h, w = mask.shape

    # Limit grid size to prevent issues with very small images
    grid_h = min(grid_h, h // 2 or 1)
    grid_w = min(grid_w, w // 2 or 1)

    # Store original mask for later use
    original_mask = mask.clone()

    # Create kernel of ones for transposed convolution
    device = mask.device
    ones = torch.ones([1, 1, grid_h, grid_w], device=device)

    # Downsample using max pooling
    pooled_mask = F.max_pool2d(mask, kernel_size=(grid_h, grid_w), stride=(grid_h, grid_w))

    # Calculate padding to ensure output has the same size as input
    output_pad_h = compute_output_padding(h, grid_h, grid_h, pooled_mask.shape[2])
    output_pad_w = compute_output_padding(w, grid_w, grid_w, pooled_mask.shape[3])
    output_padding = (output_pad_h, output_pad_w)

    # Upsample using transposed convolution
    expanded_mask = F.conv_transpose2d(
        pooled_mask,
        weight=ones,
        stride=(grid_h, grid_w),
        output_padding=output_padding
    )

    # Apply mask union operation:
    # If any pixel in the original mask is non-zero, ensure the corresponding pixel
    # in the expanded mask is also non-zero
    mask_union = torch.maximum(expanded_mask, original_mask)

    # Restore original shape if needed
    if reshape:
        mask_union = mask_union.squeeze(0)

    return mask_union