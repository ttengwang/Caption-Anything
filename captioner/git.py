from transformers import GitProcessor, AutoProcessor
from .modeling_git import GitForCausalLM
from PIL import Image
import torch
from .base_captioner import BaseCaptioner
import numpy as np
from typing import Union
import torchvision.transforms.functional as F


class GITCaptioner(BaseCaptioner):
    def __init__(self, device, enable_filter=False):
        super().__init__(device, enable_filter)
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.processor = AutoProcessor.from_pretrained("microsoft/git-large")
        self.model = GitForCausalLM.from_pretrained("microsoft/git-large", torch_dtype=self.torch_dtype).to(self.device)
    
    @torch.no_grad()
    def inference(self, image: Union[np.ndarray, Image.Image, str], filter=False):
        if type(image) == str: # input path
            image = Image.open(image)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device, self.torch_dtype)
        generated_ids = self.model.generate(pixel_values=pixel_values, max_new_tokens=50)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        if self.enable_filter and filter:
            captions = self.filter_caption(image, captions)
        print(f"\nProcessed ImageCaptioning by GITCaptioner, Output Text: {generated_caption}")
        return generated_caption

    @torch.no_grad()
    def inference_with_reduced_tokens(self, image: Union[np.ndarray, Image.Image, str], seg_mask, crop_mode="w_bg", filter=False, disable_regular_box = False):
        crop_save_path = self.generate_seg_cropped_image(image=image, seg_mask=seg_mask, crop_mode=crop_mode, disable_regular_box=disable_regular_box)
        if type(image) == str: # input path
            image = Image.open(image)
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device, self.torch_dtype)
        _, _, H, W = pixel_values.shape
        seg_mask = Image.fromarray(seg_mask.astype(float))
        seg_mask = seg_mask.resize((H, W))
        seg_mask = F.pil_to_tensor(seg_mask) > 0.5
        seg_mask = seg_mask.float()
        pixel_masks = seg_mask.unsqueeze(0).to(self.device)
        out = self.model.generate(pixel_values=pixel_values, pixel_masks=pixel_masks, max_new_tokens=50)
        captions = self.processor.decode(out[0], skip_special_tokens=True).strip()
        if self.enable_filter and filter:
            captions = self.filter_caption(image, captions)
        print(f"\nProcessed ImageCaptioning by BLIPCaptioner, Output Text: {captions}")
        return captions, crop_save_path

if __name__ == '__main__':
    model = GITCaptioner(device='cuda:2', enable_filter=False)
    image_path = 'test_img/img2.jpg'
    seg_mask = np.zeros((224,224))
    seg_mask[50:200, 50:200] = 1
    print(f'process image {image_path}')
    print(model.inference_with_reduced_tokens(image_path, seg_mask))