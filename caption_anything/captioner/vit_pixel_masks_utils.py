
import torch
import torch.nn as nn


class ViTPatchMaskGenerator(nn.Module):
    def __init__(self, patch_size) -> None:
        super(ViTPatchMaskGenerator, self).__init__()
        self.patch_size = patch_size
        self.pool = nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_masks):
        patch_mask = self.pool(pixel_masks)
        patch_mask = patch_mask.bool().flatten(1)
        cls_token_mask = patch_mask.new_ones([patch_mask.shape[0], 1]).bool()
        patch_mask = torch.cat([cls_token_mask, patch_mask], dim=-1)
        return patch_mask
